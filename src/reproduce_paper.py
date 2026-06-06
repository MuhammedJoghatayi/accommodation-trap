"""
reproduce_paper.py — Numerical pipeline for "The Accommodation Trap."

Runs every analysis in the paper and saves a single results object to
results/results.pkl. The companion script generate_figures.py reads that
object and produces every main and supplementary figure.

Usage
-----
    python reproduce_paper.py

Output
------
    results/results.pkl   — all numerical results
    results/results.log   — captured stdout
    results/metadata.json — Python/NumPy versions, runtime, timestamp


Reproducibility
---------------
    Bit-for-bit reproducibility requires matching NumPy version (see
    requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Force UTF-8 output for Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# =============================================================================
# Core model
# =============================================================================

class PlasticBeliefNetwork:
    """Symmetric Hopfield network with optional Hebbian plasticity.

    Args:
    ----------
    N : int
        Number of binary nodes.
    num_patterns : int
        Number of stored Hebbian patterns. Pattern 0 is the healthy reference.
    seed : int
        RNG seed for pattern construction and stochastic dynamics.
    weight_bound : {"max_norm", "weight_decay"}
        Bounding scheme applied to W after each plasticity increment.
        "max_norm" rescales W so its maximum absolute entry matches the
        original ceiling (main paper). "weight_decay" applies
        W ← (1 − λ)W + ΔW with no rescaling (robustness check).
    weight_decay_lambda : float
        Decay coefficient used only when weight_bound == "weight_decay".
    """

    def __init__(self, N=50, num_patterns=3, seed=0,
                 weight_bound="max_norm", weight_decay_lambda=0.001):
        self.N = N
        self.num_patterns = num_patterns
        self.seed = seed
        self.weight_bound = weight_bound
        self.weight_decay_lambda = weight_decay_lambda
        self.rng = np.random.RandomState(seed)

        self.patterns = self.rng.choice([-1, 1], size=(num_patterns, N))
        self.healthy_pattern = self.patterns[0].copy()
        self.W = np.zeros((N, N))
        for p in self.patterns:
            self.W += np.outer(p, p)
        self.W /= N
        np.fill_diagonal(self.W, 0)
        self.W_original = self.W.copy()
        self._original_max = float(np.max(np.abs(self.W_original)))

    def overlap(self, state, pattern):
        return float(np.dot(state, pattern) / self.N)

    def energy(self, state):
        return float(-0.5 * state @ self.W @ state)

    def _bound_weights(self, delta_W):
        """Apply the plasticity increment with the selected bounding scheme."""
        np.fill_diagonal(delta_W, 0)
        if self.weight_bound == "max_norm":
            self.W += delta_W
            norm = float(np.max(np.abs(self.W)))
            if norm > 0:
                self.W *= self._original_max / norm
        elif self.weight_bound == "weight_decay":
            self.W = (1.0 - self.weight_decay_lambda) * self.W + delta_W
        else:
            raise ValueError(f"Unknown weight_bound: {self.weight_bound}")
        np.fill_diagonal(self.W, 0)

    def update_with_plasticity(self, state, clamped_nodes=None,
                               learning_rate=0.0, noise_std=0.0,
                               n_steps=None, track_every=100):
        """Asynchronous update with optional Hebbian plasticity and field noise.

        Plasticity is evaluated at N-step intervals during the asynchronous
        update loop. Tied local fields (h_i == 0) are resolved to +1.
        """
        if clamped_nodes is None:
            clamped_nodes = {}
        if n_steps is None:
            n_steps = 20 * self.N

        state = state.copy()
        for k, v in clamped_nodes.items():
            state[k] = v

        energies = [self.energy(state)]
        overlaps = [self.overlap(state, self.healthy_pattern)]
        weight_changes = [0.0]

        for step in range(n_steps):
            i = self.rng.randint(self.N)
            if i in clamped_nodes:
                continue
            h_i = float(self.W[i] @ state)
            if noise_std > 0:
                h_i += self.rng.normal(0, noise_std)
            state[i] = 1.0 if h_i >= 0 else -1.0

            if learning_rate > 0 and step % self.N == 0:
                delta_W = learning_rate * np.outer(state, state) / self.N
                self._bound_weights(delta_W)

            if step % track_every == 0:
                energies.append(self.energy(state))
                overlaps.append(self.overlap(state, self.healthy_pattern))
                weight_changes.append(
                    float(np.linalg.norm(self.W - self.W_original))
                )

        return state, np.array(energies), np.array(overlaps), np.array(weight_changes)


# =============================================================================
# Common protocols
# =============================================================================

def settle(net, state, steps_mult=50):
    """Free settling: no clamps, no plasticity, no noise."""
    s, _, _, _ = net.update_with_plasticity(
        state, clamped_nodes={}, learning_rate=0.0,
        n_steps=steps_mult * net.N
    )
    return s


def accommodate(N, eta, alpha, seed, delusional_node=0, num_patterns=3,
                weight_bound="max_norm", weight_decay_lambda=0.001):
    """Run focal-coherent accommodation: clamp one node, run plasticity."""
    net = PlasticBeliefNetwork(
        N=N, num_patterns=num_patterns, seed=seed,
        weight_bound=weight_bound, weight_decay_lambda=weight_decay_lambda
    )
    dn = delusional_node
    dv = -net.healthy_pattern[dn]
    state = net.healthy_pattern.copy()
    state[dn] = dv
    s_acc, _, _, _ = net.update_with_plasticity(
        state, clamped_nodes={dn: dv},
        learning_rate=eta, n_steps=alpha * N
    )
    return net, s_acc, dn


def recovered(net, state, delusional_node=0, threshold=0.9):
    """Recovery requires overlap >= threshold AND clamped node restored."""
    ov = net.overlap(state, net.healthy_pattern)
    node_restored = (state[delusional_node] == net.healthy_pattern[delusional_node])
    return bool((ov >= threshold) and node_restored), ov


# =============================================================================
# Logging helpers
# =============================================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def hdr(t, char="="):
    print(f"\n{char * 70}\n{t}\n{char * 70}")


def subhdr(t):
    print(f"\n--- {t} ---")


def progress(msg):
    print(f"    {msg}")


# =============================================================================
# P1. Plasticity necessity
# =============================================================================

def analysis_1_plasticity_necessity():
    hdr("P1. PLASTICITY NECESSITY")
    n_seeds = 30
    progress(f"Running {n_seeds} seeds at η ∈ {{0.0, 0.02}}, α=100, N=50...")

    out = {}
    for eta in [0.0, 0.02]:
        recs, ovs = [], []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(N=50, eta=eta, alpha=100, seed=seed)
            s_post = settle(net, s_acc)
            rec, ov = recovered(net, s_post, dn)
            recs.append(rec); ovs.append(ov)
        out[f"eta_{eta}"] = {
            "rate": 100 * sum(recs) / n_seeds,
            "ov_mean": float(np.mean(ovs)),
            "ov_std": float(np.std(ovs)),
            "recoveries": [bool(r) for r in recs],
            "overlaps": [float(o) for o in ovs],
        }
        progress(f"η={eta}: recovery={out[f'eta_{eta}']['rate']:.0f}%, "
                 f"overlap={out[f'eta_{eta}']['ov_mean']:.4f}")
    return out


# =============================================================================
# P2. Two regimes
# =============================================================================

def analysis_2_two_regimes():
    hdr("P2. TWO REGIMES (focal-coherent vs diffuse-incoherent)")
    n_seeds = 50
    progress(f"Running {n_seeds} seeds for both regimes (η=0.03, α=100)...")

    subhdr("2.1 Focal-coherent (clamp on node 0)")
    dd_results = []
    for seed in range(n_seeds):
        net, s_acc, dn = accommodate(N=50, eta=0.03, alpha=100, seed=seed)
        ov_acc = net.overlap(s_acc, net.healthy_pattern)
        drift = float(np.linalg.norm(net.W - net.W_original))
        displacement = int(np.sum(s_acc != net.healthy_pattern))
        s_post = settle(net, s_acc)
        rec, ov_post = recovered(net, s_post, dn)
        dd_results.append({
            "seed": seed, "ov_acc": float(ov_acc), "drift": drift,
            "displacement": displacement, "ov_post": float(ov_post),
            "recovered": bool(rec),
        })
    rec_rate = 100 * sum(r["recovered"] for r in dd_results) / n_seeds
    drift_mean = float(np.mean([r["drift"] for r in dd_results]))
    drift_std = float(np.std([r["drift"] for r in dd_results]))
    progress(f"Recovery: {rec_rate:.0f}% | drift: {drift_mean:.3f}±{drift_std:.3f}")

    subhdr("2.2 Diffuse-incoherent (Gaussian field noise σ=2.0)")
    noise_results = []
    for seed in range(n_seeds):
        net = PlasticBeliefNetwork(N=50, num_patterns=3, seed=seed)
        state = net.healthy_pattern.copy()
        s_dis, _, _, _ = net.update_with_plasticity(
            state, clamped_nodes={}, learning_rate=0.03,
            noise_std=2.0, n_steps=100 * 50
        )
        ov_dis = net.overlap(s_dis, net.healthy_pattern)
        drift = float(np.linalg.norm(net.W - net.W_original))
        displacement = int(np.sum(s_dis != net.healthy_pattern))
        s_post = settle(net, s_dis)
        rec, ov_post = recovered(net, s_post, delusional_node=0)
        noise_results.append({
            "seed": seed, "ov_dis": float(ov_dis), "drift": drift,
            "displacement": displacement, "ov_post": float(ov_post),
            "recovered": bool(rec),
        })
    rec_rate_n = 100 * sum(r["recovered"] for r in noise_results) / n_seeds
    progress(f"Recovery: {rec_rate_n:.0f}%")

    pos = [r for r in noise_results if r["ov_dis"] > 0]
    neg = [r for r in noise_results if r["ov_dis"] <= 0]
    rec_pos = 100 * sum(r["recovered"] for r in pos) / max(len(pos), 1)
    rec_neg = 100 * sum(r["recovered"] for r in neg) / max(len(neg), 1)
    progress(f"Trajectory-dependence: pos-end → {rec_pos:.0f}% ({len(pos)} seeds), "
             f"non-pos-end → {rec_neg:.0f}% ({len(neg)} seeds)")

    return {"dd": dd_results, "noise": noise_results}


# =============================================================================
# P3. Phase-threshold sweep
# =============================================================================

def analysis_3_phase_transition():
    hdr("P3. PHASE-THRESHOLD SWEEP")
    n_seeds = 20
    eta_values = np.linspace(0.0, 0.08, 20)
    progress(f"20 η × {n_seeds} seeds at N=50, α=100...")

    results = []
    for eta in eta_values:
        persist = 0
        overlaps, drifts = [], []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(N=50, eta=float(eta), alpha=100, seed=seed)
            s_post = settle(net, s_acc)
            if s_post[dn] == -net.healthy_pattern[dn]:
                persist += 1
            overlaps.append(net.overlap(s_post, net.healthy_pattern))
            drifts.append(float(np.linalg.norm(net.W - net.W_original)))
        row = {
            "eta": float(eta),
            "persist_pct": 100 * persist / n_seeds,
            "ov_mean": float(np.mean(overlaps)),
            "drift_mean": float(np.mean(drifts)),
        }
        results.append(row)
        progress(f"η={eta:.4f}: persist={row['persist_pct']:5.1f}%, "
                 f"drift={row['drift_mean']:.3f}")
    return results


# =============================================================================
# P4. Weight-swap causal control
# =============================================================================

def analysis_4_weight_swap():
    hdr("P4. WEIGHT-SWAP CAUSAL CONTROL")
    n_seeds = 30
    progress(f"{n_seeds} seeds at η=0.03, α=100, N=50...")

    rec_orig = rec_plast = 0
    for seed in range(n_seeds):
        net, s_acc, dn = accommodate(N=50, eta=0.03, alpha=100, seed=seed)
        W_plastic = net.W.copy()
        net.W = net.W_original.copy()
        s_post = settle(net, s_acc)
        if recovered(net, s_post, dn)[0]:
            rec_orig += 1
        net.W = W_plastic
        s_post = settle(net, s_acc)
        if recovered(net, s_post, dn)[0]:
            rec_plast += 1
    progress(f"W_original: {rec_orig}/{n_seeds} recover | "
             f"W_plastic: {rec_plast}/{n_seeds} recover")
    return {
        "rec_with_orig_pct": 100 * rec_orig / n_seeds,
        "rec_with_plastic_pct": 100 * rec_plast / n_seeds,
        "n_seeds": n_seeds,
    }


# =============================================================================
# P5. Distributed accommodation
# =============================================================================

def analysis_5_distributed():
    hdr("P5. DISTRIBUTED ACCOMMODATION")
    n_seeds = 30
    progress(f"{n_seeds} seeds at η=0.02, α=100, N=50...")

    cvs, r_dW, r_dh = [], [], []
    for seed in range(n_seeds):
        net, s_acc, dn = accommodate(N=50, eta=0.02, alpha=100, seed=seed)
        dW = net.W - net.W_original
        delta_h = dW @ s_acc
        dW_per_node = np.sum(np.abs(dW), axis=1)
        conn_per_node = np.sum(np.abs(net.W_original), axis=1)
        mask = np.ones(net.N, dtype=bool); mask[dn] = False
        cvs.append(float(np.std(dW_per_node[mask]) / np.mean(dW_per_node[mask])))
        r_dW.append(float(np.corrcoef(conn_per_node[mask], dW_per_node[mask])[0, 1]))
        r_dh.append(float(np.corrcoef(conn_per_node[mask], delta_h[mask])[0, 1]))

    out = {
        "cv_mean": float(np.mean(cvs)), "cv_std": float(np.std(cvs)),
        "r_dW_mean": float(np.mean(r_dW)), "r_dW_std": float(np.std(r_dW)),
        "r_dh_mean": float(np.mean(r_dh)), "r_dh_std": float(np.std(r_dh)),
    }
    progress(f"CV: {out['cv_mean']:.3f}±{out['cv_std']:.3f}")
    progress(f"r(connectivity, ΔW): {out['r_dW_mean']:.3f}±{out['r_dW_std']:.3f}")
    progress(f"r(connectivity, Δh): {out['r_dh_mean']:.3f}±{out['r_dh_std']:.3f}")
    return out


# =============================================================================
# P6. Escape-threshold heatmap
# =============================================================================

def analysis_6_heatmap():
    hdr("P6. ESCAPE-THRESHOLD HEATMAP")
    eta_grid = np.linspace(0.0, 0.04, 12)
    alpha_grid = np.linspace(10, 200, 12).astype(int)
    n_seeds, n_trials = 8, 50
    p_values = np.concatenate([
        np.linspace(0.0, 0.2, 20),
        np.linspace(0.25, 0.5, 6),
    ])
    progress(f"{len(eta_grid)}×{len(alpha_grid)} grid, "
             f"{n_seeds} seeds, {n_trials} trials/cell, {len(p_values)} p-values")
    progress("This is the slowest analysis — expect 1–3 hours.")

    nan_frac = np.zeros((len(alpha_grid), len(eta_grid)))
    p_escape_median = np.full((len(alpha_grid), len(eta_grid)), np.nan)
    p_escape_all = np.full((len(alpha_grid), len(eta_grid), n_seeds), np.nan)

    t0 = time.time()
    n_cells = len(alpha_grid) * len(eta_grid)
    cells_done = 0

    for ai, alpha in enumerate(alpha_grid):
        for ei, eta in enumerate(eta_grid):
            seed_pe = []
            for seed in range(n_seeds):
                net, s_acc, dn = accommodate(N=50, eta=float(eta),
                                              alpha=int(alpha), seed=seed)
                pe = np.nan
                pert_rng = np.random.RandomState(seed + 5000)
                for p in p_values:
                    rec_count = 0
                    for _ in range(n_trials):
                        s = s_acc.copy()
                        flips = pert_rng.rand(net.N) < p
                        s[flips] *= -1
                        s_out = settle(net, s)
                        rec, _ = recovered(net, s_out, dn)
                        if rec:
                            rec_count += 1
                    if rec_count / n_trials >= 0.5:
                        pe = float(p)
                        break
                seed_pe.append(pe)
                p_escape_all[ai, ei, seed] = pe

            arr = np.array(seed_pe, dtype=float)
            nan_count = int(np.sum(np.isnan(arr)))
            nan_frac[ai, ei] = nan_count / n_seeds
            if nan_count < n_seeds:
                p_escape_median[ai, ei] = float(np.nanmedian(arr))

            cells_done += 1
            elapsed = time.time() - t0
            est_remaining = (elapsed / cells_done) * (n_cells - cells_done)
            progress(f"  cell {cells_done}/{n_cells} (η={eta:.4f}, α={alpha}): "
                     f"NaN={nan_count}/{n_seeds} | "
                     f"elapsed {elapsed/60:.1f}min, "
                     f"est remaining {est_remaining/60:.1f}min")

    return {
        "eta_grid": eta_grid, "alpha_grid": alpha_grid, "p_values": p_values,
        "nan_frac": nan_frac, "p_escape_median": p_escape_median,
        "p_escape_all": p_escape_all,
        "n_seeds": n_seeds, "n_trials": n_trials,
    }


# =============================================================================
# P7. Perturbation taxonomy (secondary analysis)
# =============================================================================

INTERVENTION_PLASTICITY_RATIO = 0.5
STRESS_AMPLIFICATION_FACTOR = 2.0


def _perturb_untargeted(state, p, rng):
    """Mode A: untargeted random flips."""
    s = state.copy()
    flips = rng.rand(len(s)) < p
    s[flips] *= -1
    return s


def _perturb_peripheral(state, p, healthy_pattern, dn, rng):
    """Mode B: peripheral restorative cueing (non-clamped nodes set to ξ¹)."""
    s = state.copy()
    cue = rng.rand(len(s)) < p
    cue[dn] = False
    s[cue] = healthy_pattern[cue]
    return s


def _perturb_focal(state, p, healthy_pattern, dn, rng):
    """Mode C: focal corrective cueing (clamped node set to ξ¹)."""
    s = state.copy()
    if rng.rand() < p:
        s[dn] = healthy_pattern[dn]
    return s


def _settle_with_plasticity(net, state, eta_int, n_sessions):
    """Iterated sessions with optional plasticity."""
    s = state.copy()
    for _ in range(n_sessions):
        s, _, _, _ = net.update_with_plasticity(
            s, clamped_nodes={}, learning_rate=eta_int,
            n_steps=50 * net.N
        )
    return s


def analysis_7_perturbation_taxonomy():
    hdr("P7. PERTURBATION TAXONOMY (secondary analysis)")
    n_seeds = 15
    n_trials = 30
    p_values = np.linspace(0.0, 0.5, 11)
    eta_grid = [0.005, 0.007, 0.010, 0.015, 0.020]
    alpha = 80
    N = 50
    progress(f"{len(eta_grid)} accommodation depths × {len(p_values)} p-values "
             f"× {n_seeds} seeds × {n_trials} trials")

    results = {}
    for eta_acc in eta_grid:
        eta_int = eta_acc * INTERVENTION_PLASTICITY_RATIO
        subhdr(f"η_acc={eta_acc} (η_int={eta_int:.4f})")

        accommodated = []
        n_trapped = 0
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(N=N, eta=eta_acc, alpha=alpha, seed=seed)
            accommodated.append((net, s_acc, dn))
            s_free = settle(net, s_acc)
            if s_free[dn] == -net.healthy_pattern[dn]:
                n_trapped += 1
        progress(f"{n_trapped}/{n_seeds} seeds trap under free settle")

        rates = {lbl: np.zeros(len(p_values))
                 for lbl in ["A", "B", "C", "D", "Cstress"]}

        for pi, p in enumerate(p_values):
            counts = {lbl: 0 for lbl in rates}
            for seed_idx, (net_orig, s_acc, dn) in enumerate(accommodated):
                rng = np.random.RandomState(seed_idx + 1000 + int(p * 100))

                # Each mode runs n_trials with fresh W copy each trial.
                W_acc = net_orig.W.copy()
                hp = net_orig.healthy_pattern

                for _ in range(n_trials):
                    s0 = _perturb_untargeted(s_acc, p, rng)
                    net_copy = PlasticBeliefNetwork(N=N, num_patterns=3, seed=seed_idx)
                    net_copy.W = W_acc.copy()
                    s_out = settle(net_copy, s0)
                    if recovered(net_copy, s_out, dn)[0]:
                        counts["A"] += 1

                for _ in range(n_trials):
                    s0 = _perturb_peripheral(s_acc, p, hp, dn, rng)
                    net_copy = PlasticBeliefNetwork(N=N, num_patterns=3, seed=seed_idx)
                    net_copy.W = W_acc.copy()
                    s_out = _settle_with_plasticity(net_copy, s0, eta_int, 3)
                    if recovered(net_copy, s_out, dn)[0]:
                        counts["B"] += 1

                for _ in range(n_trials):
                    s0 = _perturb_focal(s_acc, p, hp, dn, rng)
                    net_copy = PlasticBeliefNetwork(N=N, num_patterns=3, seed=seed_idx)
                    net_copy.W = W_acc.copy()
                    s_out = _settle_with_plasticity(net_copy, s0, eta_int, 3)
                    if recovered(net_copy, s_out, dn)[0]:
                        counts["C"] += 1

                for _ in range(n_trials):
                    s0 = _perturb_focal(s_acc, p, hp, dn, rng)
                    net_copy = PlasticBeliefNetwork(N=N, num_patterns=3, seed=seed_idx)
                    net_copy.W = W_acc.copy()
                    s_out = _settle_with_plasticity(net_copy, s0, 0.0, 3)
                    if recovered(net_copy, s_out, dn)[0]:
                        counts["D"] += 1

                for _ in range(n_trials):
                    s0 = _perturb_focal(s_acc, p, hp, dn, rng)
                    net_copy = PlasticBeliefNetwork(N=N, num_patterns=3, seed=seed_idx)
                    net_copy.W = W_acc.copy()
                    s_out = _settle_with_plasticity(
                        net_copy, s0,
                        eta_int * STRESS_AMPLIFICATION_FACTOR, 3
                    )
                    if recovered(net_copy, s_out, dn)[0]:
                        counts["Cstress"] += 1

            denom = n_seeds * n_trials
            for lbl in rates:
                rates[lbl][pi] = 100 * counts[lbl] / denom

        results[eta_acc] = {
            "p_values": p_values,
            "eta_int": eta_int,
            "n_sessions": 3,
            "stress_factor": STRESS_AMPLIFICATION_FACTOR,
            "n_trapped": n_trapped,
            "rates": rates,
        }
    return results


# =============================================================================
# P8. Scaling: phase transition + 1−2/N + CV across N
# =============================================================================

def analysis_8_scaling():
    hdr("P8. SCALING ACROSS NETWORK SIZE")
    Ns = [50, 100, 200]
    n_seeds = 10

    subhdr("8A. Phase transition at N ∈ {50, 100, 200}")
    eta_test = [0.005, 0.007, 0.008, 0.010, 0.020, 0.040, 0.080]
    pt = {}
    for N in Ns:
        pt[N] = []
        for eta in eta_test:
            persist = 0
            ovs = []
            for seed in range(n_seeds):
                net, s_acc, dn = accommodate(N=N, eta=eta, alpha=100, seed=seed)
                s_post = settle(net, s_acc)
                if s_post[dn] == -net.healthy_pattern[dn]:
                    persist += 1
                ovs.append(net.overlap(s_post, net.healthy_pattern))
            pt[N].append({
                "eta": eta,
                "persist_pct": 100 * persist / n_seeds,
                "ov_mean": float(np.mean(ovs)),
            })
        progress(f"N={N}: " + ", ".join(
            f"η={r['eta']}→{r['persist_pct']:.0f}%" for r in pt[N]))

    subhdr("8B. 1−2/N overlap relation (η=0.05, α=100)")
    overlap_law = {}
    for N in Ns:
        ovs, drifts = [], []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(N=N, eta=0.05, alpha=100, seed=seed)
            s_post = settle(net, s_acc)
            ovs.append(net.overlap(s_post, net.healthy_pattern))
            drifts.append(float(np.linalg.norm(net.W - net.W_original)))
        overlap_law[N] = {
            "ov_mean": float(np.mean(ovs)),
            "predicted_1_minus_2_over_N": 1 - 2 / N,
            "drift_mean": float(np.mean(drifts)),
        }
        progress(f"N={N}: overlap={overlap_law[N]['ov_mean']:.4f} "
                 f"(predicted {overlap_law[N]['predicted_1_minus_2_over_N']:.4f})")

    subhdr("8C. Distributed-accommodation CV across N (η=0.02, α=100)")
    cv = {}
    for N in Ns:
        cvs, rs = [], []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(N=N, eta=0.02, alpha=100, seed=seed)
            dW = net.W - net.W_original
            dW_per_node = np.sum(np.abs(dW), axis=1)
            conn_per_node = np.sum(np.abs(net.W_original), axis=1)
            mask = np.ones(N, dtype=bool); mask[dn] = False
            cvs.append(float(np.std(dW_per_node[mask]) / np.mean(dW_per_node[mask])))
            rs.append(float(np.corrcoef(conn_per_node[mask], dW_per_node[mask])[0, 1]))
        cv[N] = {
            "cv_mean": float(np.mean(cvs)), "cv_std": float(np.std(cvs)),
            "r_mean": float(np.mean(rs)), "r_std": float(np.std(rs)),
        }
        progress(f"N={N}: CV={cv[N]['cv_mean']:.3f}±{cv[N]['cv_std']:.3f}, "
                 f"r={cv[N]['r_mean']:.3f}±{cv[N]['r_std']:.3f}")

    return {"phase_transition": pt, "overlap_law": overlap_law, "cv": cv}


# =============================================================================
# P9. Robustness: choice of clamped node
# =============================================================================

def analysis_robustness_node_choice():
    hdr("P9. ROBUSTNESS — choice of clamped node")
    n_seeds = 10
    nodes = [0, 1, 5, 25, 49]
    progress(f"Testing nodes {nodes} × {n_seeds} seeds at η=0.03, α=100, N=50...")

    out = {}
    for node in nodes:
        persist = 0
        ovs = []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(N=50, eta=0.03, alpha=100,
                                          seed=seed, delusional_node=node)
            s_post = settle(net, s_acc)
            if s_post[dn] == -net.healthy_pattern[dn]:
                persist += 1
            ovs.append(net.overlap(s_post, net.healthy_pattern))
        out[node] = {
            "ov_mean": float(np.mean(ovs)),
            "ov_std": float(np.std(ovs)),
            "persist_pct": 100 * persist / n_seeds,
        }
        progress(f"node={node}: persist={out[node]['persist_pct']:.0f}%, "
                 f"overlap={out[node]['ov_mean']:.4f}")
    return out


# =============================================================================
# P10. Weight-decay normalization robustness
# =============================================================================

def analysis_10_weight_decay():
    hdr("P10. WEIGHT-DECAY NORMALIZATION ROBUSTNESS")
    LAMBDA = 0.001
    progress(f"Re-running key analyses with weight-decay normalization (λ={LAMBDA})")

    subhdr("10A. Plasticity necessity")
    pn = {}
    for eta in [0.0, 0.02]:
        recs, ovs = [], []
        for seed in range(30):
            net, s_acc, dn = accommodate(
                N=50, eta=eta, alpha=100, seed=seed,
                weight_bound="weight_decay", weight_decay_lambda=LAMBDA
            )
            s_post = settle(net, s_acc)
            rec, ov = recovered(net, s_post, dn)
            recs.append(rec); ovs.append(ov)
        pn[eta] = {
            "rate": 100 * sum(recs) / 30,
            "ov_mean": float(np.mean(ovs)),
            "ov_std": float(np.std(ovs)),
        }
        progress(f"η={eta}: recovery={pn[eta]['rate']:.0f}%")

    subhdr("10B. Phase-threshold sweep")
    pt = []
    for eta in np.linspace(0.0, 0.08, 20):
        persist = 0
        ovs, drifts = [], []
        for seed in range(20):
            net, s_acc, dn = accommodate(
                N=50, eta=float(eta), alpha=100, seed=seed,
                weight_bound="weight_decay", weight_decay_lambda=LAMBDA
            )
            s_post = settle(net, s_acc)
            if s_post[dn] == -net.healthy_pattern[dn]:
                persist += 1
            ovs.append(net.overlap(s_post, net.healthy_pattern))
            drifts.append(float(np.linalg.norm(net.W - net.W_original)))
        pt.append({
            "eta": float(eta),
            "persist_pct": 100 * persist / 20,
            "ov_mean": float(np.mean(ovs)),
            "drift_mean": float(np.mean(drifts)),
        })

    subhdr("10C. Weight-swap")
    rec_orig = rec_plast = 0
    for seed in range(30):
        net, s_acc, dn = accommodate(
            N=50, eta=0.03, alpha=100, seed=seed,
            weight_bound="weight_decay", weight_decay_lambda=LAMBDA
        )
        W_plastic = net.W.copy()
        net.W = net.W_original.copy()
        s_post = settle(net, s_acc)
        if recovered(net, s_post, dn)[0]: rec_orig += 1
        net.W = W_plastic
        s_post = settle(net, s_acc)
        if recovered(net, s_post, dn)[0]: rec_plast += 1
    ws = {"rec_with_orig_pct": 100 * rec_orig / 30,
          "rec_with_plastic_pct": 100 * rec_plast / 30}
    progress(f"W_original: {ws['rec_with_orig_pct']:.0f}% | "
             f"W_plastic: {ws['rec_with_plastic_pct']:.0f}%")

    subhdr("10D. 1−2/N relation")
    law = {}
    for N in [50, 100, 200]:
        ovs, drifts = [], []
        for seed in range(10):
            net, s_acc, dn = accommodate(
                N=N, eta=0.05, alpha=100, seed=seed,
                weight_bound="weight_decay", weight_decay_lambda=LAMBDA
            )
            s_post = settle(net, s_acc)
            ovs.append(net.overlap(s_post, net.healthy_pattern))
            drifts.append(float(np.linalg.norm(net.W - net.W_original)))
        law[N] = {
            "ov_mean": float(np.mean(ovs)),
            "predicted_1_minus_2_over_N": 1 - 2 / N,
            "drift_mean": float(np.mean(drifts)),
        }
        progress(f"N={N}: overlap={law[N]['ov_mean']:.4f}")

    return {
        "weight_decay_lambda": LAMBDA,
        "plasticity_necessity_wd": pn,
        "phase_transition_wd": pt,
        "weight_swap_wd": ws,
        "scaling_2_over_N_wd": law,
    }


# =============================================================================
# P11. Weight interpolation (structural reversibility)
# =============================================================================

def analysis_11_weight_interpolation():
    hdr("P11. STRUCTURAL INTERPOLATION (W_γ = (1−γ)W_plastic + γW_original)")
    n_seeds = 30
    eta_acc = 0.03
    alpha_acc = 100
    gammas = np.linspace(0.0, 1.0, 11)
    progress(f"{len(gammas)} γ × {n_seeds} seeds at η_acc=0.03, α_acc=100, N=50...")

    rows = []
    for gamma in gammas:
        recs, ovs, node_restored = [], [], []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(N=50, eta=eta_acc, alpha=alpha_acc, seed=seed)
            W_plastic = net.W.copy()
            W_original = net.W_original.copy()
            net.W = (1.0 - gamma) * W_plastic + gamma * W_original
            np.fill_diagonal(net.W, 0)

            s_post = settle(net, s_acc)
            rec, ov = recovered(net, s_post, dn)
            recs.append(rec); ovs.append(ov)
            node_restored.append(s_post[dn] == net.healthy_pattern[dn])

        row = {
            "gamma": float(gamma),
            "recovery_pct": 100 * np.mean(recs),
            "ov_mean": float(np.mean(ovs)),
            "ov_std": float(np.std(ovs)),
            "node_restored_pct": 100 * np.mean(node_restored),
            "n_seeds": n_seeds,
        }
        rows.append(row)
        progress(f"γ={gamma:.2f}: recovery={row['recovery_pct']:5.1f}%, "
                 f"overlap={row['ov_mean']:.3f}±{row['ov_std']:.3f}")

    return {
        "params": {
            "N": 50, "eta_acc": eta_acc, "alpha_acc": alpha_acc,
            "n_seeds": n_seeds, "gammas": list(gammas),
        },
        "rows": rows,
    }


# =============================================================================
# P12. Capacity test (P/N sweep)
# =============================================================================

def analysis_12_capacity():
    """Run the principal findings across multiple memory loads (P/N).

    Tests whether attractor accommodation is specific to sub-capacity regimes
    or generalizes toward Hopfield capacity (P/N ≈ 0.138). At each load, we
    report:
      - Baseline recoverability (η = 0, no plasticity): can the network
        retrieve ξ⁽¹⁾ at all at this load?
      - Accommodated recoverability (η = 0.02 and 0.03): does plasticity-driven
        accommodation still produce persistence above the baseline interference?
      - Weight-swap dissociation at η = 0.03.
      - 1 − 2/N relation at η = 0.05 above threshold.

    The contrast between baseline and accommodated is the honest measure of
    whether accommodation is additional to baseline interference at high load.
    """
    hdr("P12. CAPACITY TEST (P/N sweep)")
    N = 50
    n_seeds = 30
    # P values chosen so P/N ∈ {0.06, 0.10, 0.14} (the 0.14 case is just above
    # the asymptotic Hopfield capacity ≈ 0.138 and tests near-capacity dynamics).
    Ps = [3, 5, 7]
    progress(f"P values {Ps} (P/N = {[round(p/N, 3) for p in Ps]}) at N=50, {n_seeds} seeds...")

    out = {"N": N, "n_seeds": n_seeds, "Ps": Ps}

    for P in Ps:
        ratio = P / N
        subhdr(f"P = {P}  (P/N = {ratio:.3f})")
        load_results = {"P": P, "ratio": ratio}

        # 12A — Baseline recoverability (η = 0, no plasticity)
        baseline_recs, baseline_ovs = [], []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(
                N=N, eta=0.0, alpha=100, seed=seed, num_patterns=P
            )
            s_post = settle(net, s_acc)
            rec, ov = recovered(net, s_post, dn)
            baseline_recs.append(rec)
            baseline_ovs.append(ov)
        load_results["baseline"] = {
            "rate": 100 * sum(baseline_recs) / n_seeds,
            "ov_mean": float(np.mean(baseline_ovs)),
            "ov_std": float(np.std(baseline_ovs)),
        }
        progress(f"  Baseline (η=0):  recovery={load_results['baseline']['rate']:.0f}%, "
                 f"overlap={load_results['baseline']['ov_mean']:.4f}")

        # 12B — Accommodated recoverability at η = 0.02
        acc_recs_02, acc_ovs_02 = [], []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(
                N=N, eta=0.02, alpha=100, seed=seed, num_patterns=P
            )
            s_post = settle(net, s_acc)
            rec, ov = recovered(net, s_post, dn)
            acc_recs_02.append(rec)
            acc_ovs_02.append(ov)
        load_results["accommodated_eta_002"] = {
            "rate": 100 * sum(acc_recs_02) / n_seeds,
            "ov_mean": float(np.mean(acc_ovs_02)),
            "ov_std": float(np.std(acc_ovs_02)),
        }
        progress(f"  η=0.02:          recovery={load_results['accommodated_eta_002']['rate']:.0f}%, "
                 f"overlap={load_results['accommodated_eta_002']['ov_mean']:.4f}")

        # 12C — Accommodated recoverability at η = 0.03 (the canonical condition)
        acc_recs_03, acc_ovs_03 = [], []
        drifts_03 = []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(
                N=N, eta=0.03, alpha=100, seed=seed, num_patterns=P
            )
            drifts_03.append(float(np.linalg.norm(net.W - net.W_original)))
            s_post = settle(net, s_acc)
            rec, ov = recovered(net, s_post, dn)
            acc_recs_03.append(rec)
            acc_ovs_03.append(ov)
        load_results["accommodated_eta_003"] = {
            "rate": 100 * sum(acc_recs_03) / n_seeds,
            "ov_mean": float(np.mean(acc_ovs_03)),
            "ov_std": float(np.std(acc_ovs_03)),
            "drift_mean": float(np.mean(drifts_03)),
            "drift_std": float(np.std(drifts_03)),
        }
        progress(f"  η=0.03:          recovery={load_results['accommodated_eta_003']['rate']:.0f}%, "
                 f"overlap={load_results['accommodated_eta_003']['ov_mean']:.4f}, "
                 f"drift={load_results['accommodated_eta_003']['drift_mean']:.3f}")

        # 12D — Weight-swap at η = 0.03
        rec_orig = rec_plast = 0
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(
                N=N, eta=0.03, alpha=100, seed=seed, num_patterns=P
            )
            W_plastic = net.W.copy()
            net.W = net.W_original.copy()
            s_post = settle(net, s_acc)
            if recovered(net, s_post, dn)[0]:
                rec_orig += 1
            net.W = W_plastic
            s_post = settle(net, s_acc)
            if recovered(net, s_post, dn)[0]:
                rec_plast += 1
        load_results["weight_swap"] = {
            "rec_with_orig_pct": 100 * rec_orig / n_seeds,
            "rec_with_plastic_pct": 100 * rec_plast / n_seeds,
        }
        progress(f"  Weight-swap:    W_orig={load_results['weight_swap']['rec_with_orig_pct']:.0f}%, "
                 f"W_plast={load_results['weight_swap']['rec_with_plastic_pct']:.0f}%")

        # 12E — 1 − 2/N relation at η = 0.05 (above threshold)
        overlaps_05 = []
        for seed in range(n_seeds):
            net, s_acc, dn = accommodate(
                N=N, eta=0.05, alpha=100, seed=seed, num_patterns=P
            )
            s_post = settle(net, s_acc)
            overlaps_05.append(net.overlap(s_post, net.healthy_pattern))
        load_results["one_minus_2_over_N"] = {
            "ov_mean": float(np.mean(overlaps_05)),
            "ov_std": float(np.std(overlaps_05)),
            "predicted": 1 - 2 / N,
        }
        progress(f"  1−2/N (η=0.05): overlap={load_results['one_minus_2_over_N']['ov_mean']:.4f} "
                 f"(predicted {load_results['one_minus_2_over_N']['predicted']:.4f})")

        out[f"P_{P}"] = load_results

    return out


# =============================================================================
# P13. Sequential coherent perturbation control
# =============================================================================

def analysis_13_sequential_clamping():
    """Test whether sustained focal duration α matters, not just coherence.

    Standard accommodation: clamp node 0 for the full α × N updates.
    Sequential coherent: rotate the clamped node every (α/k) × N updates,
    cycling through k distinct nodes. Total update count is held constant
    so the cumulative plasticity exposure is matched; what differs is the
    duration each individual node is held.

    Prediction: with k > 1 (sequential), no single node experiences enough
    sustained fixation to lock in. With k = 1 (canonical, no rotation),
    accommodation occurs as in the main pipeline.
    """
    hdr("P13. SEQUENTIAL COHERENT PERTURBATION CONTROL")
    N = 50
    eta = 0.03
    alpha = 100
    n_seeds = 30
    rotations = [1, 2, 4, 10]   # k = 1 is the canonical case (no rotation)
    progress(f"Testing k ∈ {rotations} rotations across {n_seeds} seeds "
             f"(N={N}, η={eta}, α={alpha}, fixed total updates = α×N)")

    out = {"N": N, "eta": eta, "alpha": alpha, "n_seeds": n_seeds, "rotations": rotations}

    for k in rotations:
        if k == 1:
            subhdr(f"k = 1 (canonical, no rotation)")
        else:
            subhdr(f"k = {k} (clamped node rotates every {alpha/k:.0f}·N updates)")

        # Per-node persistence: was each node-that-got-clamped persistently flipped?
        persist_count = 0
        ovs = []
        drifts = []
        # Per-test: track recovery using node 0 as the canonical "delusional" node
        # for k > 1, the rotation starts at node 0 and cycles forward
        rec_count = 0

        for seed in range(n_seeds):
            net = PlasticBeliefNetwork(N=N, num_patterns=3, seed=seed)
            dn0 = 0  # canonical delusional node for recovery test
            dv0 = -net.healthy_pattern[dn0]
            state = net.healthy_pattern.copy()
            state[dn0] = dv0

            # Total updates split equally across k clamping epochs
            updates_per_epoch = (alpha * N) // k
            for epoch in range(k):
                clamp_node = epoch % N  # rotates through nodes 0, 1, 2, ...
                clamp_val = -net.healthy_pattern[clamp_node]
                state[clamp_node] = clamp_val
                state, _, _, _ = net.update_with_plasticity(
                    state,
                    clamped_nodes={clamp_node: clamp_val},
                    learning_rate=eta,
                    n_steps=updates_per_epoch,
                )

            drifts.append(float(np.linalg.norm(net.W - net.W_original)))
            s_acc = state.copy()
            # Release and free-settle
            s_post = settle(net, s_acc)
            ovs.append(net.overlap(s_post, net.healthy_pattern))

            # Persistence of node 0 (the first-clamped node)
            if s_post[dn0] == dv0:
                persist_count += 1

            # Standard recovery criterion using node 0 as delusional node
            rec, _ = recovered(net, s_post, dn0)
            if rec:
                rec_count += 1

        out[f"k_{k}"] = {
            "k": k,
            "persist_pct_node0": 100 * persist_count / n_seeds,
            "recovery_pct": 100 * rec_count / n_seeds,
            "ov_mean": float(np.mean(ovs)),
            "ov_std": float(np.std(ovs)),
            "drift_mean": float(np.mean(drifts)),
            "drift_std": float(np.std(drifts)),
            "updates_per_epoch": (alpha * N) // k,
        }
        progress(f"  k={k}: node-0 persistence={out[f'k_{k}']['persist_pct_node0']:.0f}%, "
                 f"recovery={out[f'k_{k}']['recovery_pct']:.0f}%, "
                 f"overlap={out[f'k_{k}']['ov_mean']:.4f}, "
                 f"drift={out[f'k_{k}']['drift_mean']:.3f}")

    return out


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce all numerical results.")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: ./results)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "results.log"
    pkl_path = output_dir / "results.pkl"
    meta_path = output_dir / "metadata.json"

    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)

    print("=" * 70)
    print("THE ACCOMMODATION TRAP — full numerical pipeline")
    print("=" * 70)
    print(f"Started:        {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python:         {platform.python_version()}")
    print(f"NumPy:          {np.__version__}")
    print(f"Output dir:     {output_dir.resolve()}")

    t0 = time.time()
    results = {}
    results["plasticity_necessity"]    = analysis_1_plasticity_necessity()
    results["two_regimes"]             = analysis_2_two_regimes()
    results["phase_transition"]        = analysis_3_phase_transition()
    results["weight_swap"]             = analysis_4_weight_swap()
    results["distributed"]             = analysis_5_distributed()
    results["heatmap"]                 = analysis_6_heatmap()
    results["robustness_node_choice"]  = analysis_robustness_node_choice()
    results["intervention_taxonomy"]   = analysis_7_perturbation_taxonomy()
    results["scaling"]                 = analysis_8_scaling()
    results["weight_decay"]            = analysis_10_weight_decay()
    results["weight_interpolation"]    = analysis_11_weight_interpolation()
    results["capacity"]                = analysis_12_capacity()
    results["sequential_clamping"]     = analysis_13_sequential_clamping()

    runtime_hours = (time.time() - t0) / 3600
    metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runtime_hours": round(runtime_hours, 3),
    }
    results["metadata"] = metadata

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE. Total runtime: {runtime_hours:.2f} hours")
    print(f"Results saved to: {pkl_path}")
    print(f"Metadata saved to: {meta_path}")
    print(f"{'=' * 70}\n")

    log_f.close()


if __name__ == "__main__":
    main()
