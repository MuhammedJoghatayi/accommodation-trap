"""
generate_figures.py — Render all main and supplementary figures.

Reads results/results.pkl (produced by reproduce_paper.py) and writes 7
figures to figures/: 6 main figures (Fig 1–6) and 1 supplementary
(Fig S1). Each figure is written as both PDF (vector) and PNG (600 dpi).

Aesthetic policy:
    - Muted scientific palette (burgundy, dark gray, muted blue, teal).
    - No embedded panel titles, no in-figure legends as titles.
    - Panel labels (A, B, C) outside the axes.
    - Captions live in the manuscript, not in the image.

Usage
-----
    python generate_figures.py
    python generate_figures.py --results results/results.pkl --output-dir figures
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Force UTF-8 for Windows compatibility (script may print Greek letters).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# =============================================================================
# Aesthetic configuration
# =============================================================================

C_BURGUNDY  = "#8B2C3D"
C_DARKGRAY  = "#444444"
C_MUTEDBLUE = "#3A6FA0"
C_TEAL      = "#3A8B7C"
C_LIGHTGRAY = "#BFBFBF"

N_COLORS = {50: C_BURGUNDY, 100: C_MUTEDBLUE, 200: C_TEAL}

MODE_LABELS = {
    "A":       "A: untargeted random flips",
    "B":       "B: peripheral restorative cueing",
    "C":       "C: focal corrective cueing",
    "D":       "D: focal corrective cueing, plasticity-suppressed",
    "Cstress": "C*: focal corrective cueing, elevated plasticity",
}
MODE_COLORS = {
    "A":       C_BURGUNDY,
    "B":       C_MUTEDBLUE,
    "C":       C_TEAL,
    "D":       C_DARKGRAY,
    "Cstress": "#B47A1F",
}
MODE_STYLES = {
    "A":       {"linestyle": "-",  "marker": "o"},
    "B":       {"linestyle": "-",  "marker": "s"},
    "C":       {"linestyle": "-",  "marker": "^"},
    "D":       {"linestyle": "--", "marker": "v"},
    "Cstress": {"linestyle": ":",  "marker": "D"},
}


def setup_style():
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica",
                               "Liberation Sans", "Verdana"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "lines.linewidth":    1.4,
        "lines.markersize":   5,
        "figure.dpi":         450,
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def panel_label(ax, label, dx=-0.18, dy=1.04):
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


def save_both(fig, output_dir, name):
    """Save each figure as EPS (vector, journal-preferred), PDF (vector,
    convenient), and PNG (raster, 600 dpi)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.eps")
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)
    print(f"  → {output_dir / f'{name}.eps'}")
    print(f"  → {output_dir / f'{name}.pdf'}")
    print(f"  → {output_dir / f'{name}.png'}")


# =============================================================================
# Figure 1 — Phase threshold
# =============================================================================

def fig1_phase_threshold(R, output_dir):
    print("Figure 1: phase threshold...")
    rows = R["phase_transition"]
    etas        = [r["eta"]         for r in rows]
    persist_pct = [r["persist_pct"] for r in rows]
    drift_mean  = [r["drift_mean"]  for r in rows]

    fig, ax1 = plt.subplots(figsize=(5.2, 3.5))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    ax1.plot(etas, persist_pct, color=C_BURGUNDY, marker="o",
             markersize=4, linewidth=1.6)
    ax1.set_xlabel("Plasticity rate η")
    ax1.set_ylabel("Post-clamp persistence (%)", color=C_BURGUNDY)
    ax1.tick_params(axis="y", labelcolor=C_BURGUNDY)
    ax1.set_ylim(-5, 105)

    ax2.plot(etas, drift_mean, color=C_DARKGRAY, marker="s",
             markersize=3.5, linewidth=1.2, linestyle="--")
    ax2.set_ylabel("Weight drift  ‖W − W₀‖_F", color=C_DARKGRAY)
    ax2.tick_params(axis="y", labelcolor=C_DARKGRAY)
    ax2.set_ylim(bottom=0)

    save_both(fig, output_dir, "fig1_phase_threshold")


# =============================================================================
# Figure 2 — Focal-coherent vs diffuse-incoherent
# =============================================================================

def fig2_focal_vs_diffuse(R, output_dir):
    print("Figure 2: focal vs diffuse...")
    dd    = R["two_regimes"]["dd"]
    noise = R["two_regimes"]["noise"]
    dd_drift    = np.array([r["drift"]        for r in dd])
    dd_disp     = np.array([r["displacement"] for r in dd])
    noise_drift = np.array([r["drift"]        for r in noise])
    noise_disp  = np.array([r["displacement"] for r in noise])

    noise_dis   = np.array([r["ov_dis"]  for r in noise])
    noise_post  = np.array([r["ov_post"] for r in noise])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    # Panel A: weight drift
    ax = axes[0]
    bp = ax.boxplot([dd_drift, noise_drift], patch_artist=True,
                    widths=0.55, medianprops={"color": "white", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], [C_BURGUNDY, C_MUTEDBLUE]):
        patch.set_facecolor(color); patch.set_alpha(0.8); patch.set_edgecolor("none")
    for w in bp["whiskers"]: w.set_color(C_DARKGRAY)
    for c in bp["caps"]:     c.set_color(C_DARKGRAY)
    for f in bp["fliers"]:   f.set_markersize(3); f.set_markeredgecolor(C_DARKGRAY)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Focal-\ncoherent", "Diffuse-\nincoherent"])
    ax.set_ylabel("Weight drift  ‖W − W₀‖_F")
    panel_label(ax, "A")

    # Panel B: state displacement
    ax = axes[1]
    bp = ax.boxplot([dd_disp, noise_disp], patch_artist=True,
                    widths=0.55, medianprops={"color": "white", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], [C_BURGUNDY, C_MUTEDBLUE]):
        patch.set_facecolor(color); patch.set_alpha(0.8); patch.set_edgecolor("none")
    for w in bp["whiskers"]: w.set_color(C_DARKGRAY)
    for c in bp["caps"]:     c.set_color(C_DARKGRAY)
    for f in bp["fliers"]:   f.set_markersize(3); f.set_markeredgecolor(C_DARKGRAY)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Focal-\ncoherent", "Diffuse-\nincoherent"])
    ax.set_ylabel("Nodes displaced from ξ⁽¹⁾")
    panel_label(ax, "B")

    # Panel C: trajectory-dependent recovery in diffuse regime
    ax = axes[2]
    recovered = noise_post >= 0.9
    ax.scatter(noise_dis[~recovered], noise_post[~recovered],
               s=22, color=C_DARKGRAY, alpha=0.7, edgecolor="none",
               label="Not recovered")
    ax.scatter(noise_dis[recovered], noise_post[recovered],
               s=22, color=C_BURGUNDY, alpha=0.85, edgecolor="none",
               label="Recovered")
    ax.axvline(0, color=C_LIGHTGRAY, linewidth=0.8, linestyle=":")
    ax.axhline(0.9, color=C_LIGHTGRAY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Disorder-phase overlap with ξ⁽¹⁾")
    ax.set_ylabel("Post-release overlap with ξ⁽¹⁾")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "C")

    fig.tight_layout()
    save_both(fig, output_dir, "fig2_focal_vs_diffuse")


# =============================================================================
# Figure 3 — Weight-swap + interpolation
# =============================================================================

def fig3_weight_swap_and_interpolation(R, output_dir):
    print("Figure 3: weight-swap + interpolation...")
    ws = R["weight_swap"]
    rec_orig  = ws["rec_with_orig_pct"]
    rec_plast = ws["rec_with_plastic_pct"]
    n_seeds   = ws["n_seeds"]

    interp_rows = R["weight_interpolation"]["rows"]
    gammas       = np.array([row["gamma"]         for row in interp_rows])
    recovery_pct = np.array([row["recovery_pct"]  for row in interp_rows])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    # Panel A: weight-swap strip plot
    ax = axes[0]
    rng = np.random.RandomState(0)
    x_orig  = 0 + rng.uniform(-0.18, 0.18, size=n_seeds)
    y_orig  = np.ones(n_seeds)  + rng.uniform(-0.015, 0.015, size=n_seeds)
    x_plast = 1 + rng.uniform(-0.18, 0.18, size=n_seeds)
    y_plast = np.zeros(n_seeds) + rng.uniform(-0.015, 0.015, size=n_seeds)
    ax.scatter(x_orig,  y_orig,  s=28, color=C_DARKGRAY, alpha=0.65, edgecolor="none")
    ax.scatter(x_plast, y_plast, s=28, color=C_BURGUNDY, alpha=0.75, edgecolor="none")
    ax.scatter([0, 1], [rec_orig / 100, rec_plast / 100],
               s=90, color="black", marker="_", linewidths=2.5, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["W^(original)", "W^(plastic)"])
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["Not\nrecovered", "", "Recovered"])
    ax.set_ylabel("Per-seed recovery outcome")
    ax.set_xlim(-0.5, 1.5)
    panel_label(ax, "A")

    # Panel B: interpolation curve
    ax = axes[1]
    ax.plot(gammas, recovery_pct, color=C_BURGUNDY, marker="o",
            markersize=5, linewidth=1.6)
    ax.axhline(50, color=C_LIGHTGRAY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Interpolation γ  (0: plastic → 1: original)")
    ax.set_ylabel("Recovery rate (%)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-5, 105)
    panel_label(ax, "B")

    fig.tight_layout()
    save_both(fig, output_dir, "fig3_weight_swap_and_interpolation")


# =============================================================================
# Figure 4 — Heatmap
# =============================================================================

def fig4_heatmap(R, output_dir):
    print("Figure 4: heatmap...")
    hm = R["heatmap"]
    eta_grid   = hm["eta_grid"]
    alpha_grid = hm["alpha_grid"]
    nan_frac   = hm["nan_frac"]

    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "muted_red", ["#FFFFFF", "#F4D8DC", "#D88B95", "#A8455A", C_BURGUNDY], N=256
    )

    im = ax.imshow(
        nan_frac, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
        extent=[eta_grid[0], eta_grid[-1], alpha_grid[0], alpha_grid[-1]],
    )

    eta_mesh, alpha_mesh = np.meshgrid(eta_grid, alpha_grid)
    ax.contour(eta_mesh, alpha_mesh, nan_frac, levels=[0.5],
               colors=C_DARKGRAY, linewidths=1.2, linestyles="--")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Fraction of seeds with no escape")
    cbar.outline.set_linewidth(0.6)

    ax.set_xlabel("Plasticity rate η")
    ax.set_ylabel("Accommodation duration α")

    save_both(fig, output_dir, "fig4_heatmap")


# =============================================================================
# Figure 5 — Scaling across N
# =============================================================================

def fig5_scaling(R, output_dir):
    print("Figure 5: scaling across N...")
    pt  = R["scaling"]["phase_transition"]
    law = R["scaling"]["overlap_law"]
    cv  = R["scaling"]["cv"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    # Panel A: phase transition at each N
    ax = axes[0]
    for N, rows in pt.items():
        etas = [r["eta"]         for r in rows]
        pct  = [r["persist_pct"] for r in rows]
        ax.plot(etas, pct, color=N_COLORS[N], marker="o", markersize=4,
                linewidth=1.4, label=f"N = {N}")
    ax.set_xlabel("Plasticity rate η")
    ax.set_ylabel("Persistence (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "A")

    # Panel B: 1 − 2/N law
    ax = axes[1]
    Ns_sorted = sorted(law.keys())
    obs  = [law[N]["ov_mean"] for N in Ns_sorted]
    pred = [law[N]["predicted_1_minus_2_over_N"] for N in Ns_sorted]
    ax.plot(Ns_sorted, pred, color=C_LIGHTGRAY, linewidth=1.6,
            linestyle="--", label="1 − 2/N (predicted)")
    ax.scatter(Ns_sorted, obs, s=60, color=C_BURGUNDY, zorder=3,
               edgecolor="none", label="Observed")
    ax.set_xlabel("Network size N")
    ax.set_ylabel("Post-release overlap")
    ax.set_xlim(min(Ns_sorted) - 20, max(Ns_sorted) + 20)
    ax.set_ylim(0.94, 1.005)
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "B")

    # Panel C: CV of per-node ΔW across N
    ax = axes[2]
    cv_mean = [cv[N]["cv_mean"] for N in Ns_sorted]
    cv_std  = [cv[N]["cv_std"]  for N in Ns_sorted]
    ax.errorbar(Ns_sorted, cv_mean, yerr=cv_std,
                color=C_MUTEDBLUE, marker="s", markersize=6, linewidth=1.4,
                capsize=4, ecolor=C_DARKGRAY)
    ax.set_xlabel("Network size N")
    ax.set_ylabel("CV of per-node ΔW")
    ax.set_xlim(min(Ns_sorted) - 20, max(Ns_sorted) + 20)
    ax.set_ylim(0, max(cv_mean) * 1.4)
    panel_label(ax, "C")

    fig.tight_layout()
    save_both(fig, output_dir, "fig5_scaling")


# =============================================================================
# Figure 6 — Perturbation taxonomy
# =============================================================================

def fig6_perturbation_taxonomy(R, output_dir):
    print("Figure 6: perturbation taxonomy...")
    intv = R["intervention_taxonomy"]
    eta_grid = sorted(intv.keys())

    fig, axes = plt.subplots(1, len(eta_grid), figsize=(13, 3.2), sharey=True)

    for ax, eta_acc in zip(axes, eta_grid):
        d = intv[eta_acc]
        p_values = d["p_values"]
        rates    = d["rates"]
        n_trap   = d["n_trapped"]

        for mode in ["A", "B", "C", "D", "Cstress"]:
            ax.plot(p_values, rates[mode],
                    color=MODE_COLORS[mode],
                    linestyle=MODE_STYLES[mode]["linestyle"],
                    marker=MODE_STYLES[mode]["marker"],
                    markersize=3.5, linewidth=1.3,
                    label=MODE_LABELS[mode],
                    alpha=0.9)
        ax.set_xlabel("Perturbation strength p")
        ax.text(0.5, 1.10, f"η_acc = {eta_acc:.3f}\n({n_trap}/15 trapped)",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=C_DARKGRAY)
        ax.set_ylim(-5, 105)
        ax.set_xlim(-0.02, 0.52)

    axes[0].set_ylabel("Recovery rate (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=9)
    fig.tight_layout()
    save_both(fig, output_dir, "fig6_perturbation_taxonomy")


# =============================================================================
# Figure 7 — Capacity test (P/N sweep)
# =============================================================================

def fig7_capacity(R, output_dir):
    print("Figure 7: capacity test...")
    if "capacity" not in R:
        print("  SKIPPED — no 'capacity' key in results.pkl")
        return
    cap = R["capacity"]
    Ps = cap["Ps"]
    N = cap["N"]
    ratios = [P / N for P in Ps]

    # Pull out the per-load data
    baseline = [cap[f"P_{P}"]["baseline"]["rate"]                  for P in Ps]
    acc_02   = [cap[f"P_{P}"]["accommodated_eta_002"]["rate"]      for P in Ps]
    acc_03   = [cap[f"P_{P}"]["accommodated_eta_003"]["rate"]      for P in Ps]
    ws_orig  = [cap[f"P_{P}"]["weight_swap"]["rec_with_orig_pct"]  for P in Ps]
    ws_plast = [cap[f"P_{P}"]["weight_swap"]["rec_with_plastic_pct"] for P in Ps]
    one_2_N  = [cap[f"P_{P}"]["one_minus_2_over_N"]["ov_mean"]     for P in Ps]
    predicted = [cap[f"P_{P}"]["one_minus_2_over_N"]["predicted"]  for P in Ps]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

    # Panel A: Baseline vs accommodated recoverability
    ax = axes[0, 0]
    x = np.arange(len(Ps))
    w = 0.27
    ax.bar(x - w, baseline, width=w, color=C_DARKGRAY,  alpha=0.85,
           edgecolor="none", label="Baseline (η = 0)")
    ax.bar(x,     acc_02,   width=w, color="#B47A1F",   alpha=0.85,
           edgecolor="none", label="Accommodated (η = 0.02)")
    ax.bar(x + w, acc_03,   width=w, color=C_BURGUNDY,  alpha=0.85,
           edgecolor="none", label="Accommodated (η = 0.03)")
    for xi, vals in zip(x, zip(baseline, acc_02, acc_03)):
        for offset, v in zip([-w, 0, w], vals):
            ax.text(xi + offset, v + 3, f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=8, color=C_DARKGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P = {P}\nP/N = {r:.2f}" for P, r in zip(Ps, ratios)])
    ax.set_ylabel("Recovery rate (%)")
    ax.set_ylim(0, 128)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=1, frameon=False, fontsize=7.5)
    panel_label(ax, "A")

    # Panel B: Weight-swap dissociation across loads
    ax = axes[0, 1]
    ax.bar(x - w/2, ws_orig,  width=w, color=C_DARKGRAY, alpha=0.85,
           edgecolor="none", label="W^(original)")
    ax.bar(x + w/2, ws_plast, width=w, color=C_BURGUNDY, alpha=0.85,
           edgecolor="none", label="W^(plastic)")
    for xi, (vo, vp) in zip(x, zip(ws_orig, ws_plast)):
        ax.text(xi - w/2, vo + 3, f"{vo:.0f}%", ha="center", va="bottom",
                fontsize=8, color=C_DARKGRAY)
        ax.text(xi + w/2, vp + 3, f"{vp:.0f}%", ha="center", va="bottom",
                fontsize=8, color=C_DARKGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P = {P}\nP/N = {r:.2f}" for P, r in zip(Ps, ratios)])
    ax.set_ylabel("Recovery rate (%)")
    ax.set_ylim(0, 128)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, frameon=False, fontsize=7.5)
    panel_label(ax, "B")

    # Panel C: 1 − 2/N relation across loads
    ax = axes[1, 0]
    ax.plot(ratios, predicted, color=C_LIGHTGRAY, linewidth=1.6, linestyle="--",
            label="Predicted 1 − 2/N")
    ax.scatter(ratios, one_2_N, s=70, color=C_BURGUNDY, zorder=3,
               edgecolor="none", label="Observed")
    for r, v in zip(ratios, one_2_N):
        ax.text(r, v - 0.005, f"{v:.3f}", ha="center", va="top",
                fontsize=8, color=C_DARKGRAY)
    ax.set_xlabel("Memory load  P/N")
    ax.set_ylabel("Post-release overlap with ξ⁽¹⁾")
    ax.set_xlim(min(ratios) - 0.01, max(ratios) + 0.01)
    ax.set_ylim(0.93, 0.97)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    panel_label(ax, "C")

    # Panel D: Weight drift at η = 0.03 across loads (informational)
    ax = axes[1, 1]
    drifts = [cap[f"P_{P}"]["accommodated_eta_003"]["drift_mean"] for P in Ps]
    drift_sd = [cap[f"P_{P}"]["accommodated_eta_003"]["drift_std"] for P in Ps]
    ax.errorbar(ratios, drifts, yerr=drift_sd,
                color=C_MUTEDBLUE, marker="s", markersize=6, linewidth=1.4,
                capsize=4, ecolor=C_DARKGRAY)
    for r, d in zip(ratios, drifts):
        ax.text(r, d + 0.05, f"{d:.2f}", ha="center", va="bottom",
                fontsize=8, color=C_DARKGRAY)
    ax.set_xlabel("Memory load  P/N")
    ax.set_ylabel("Frobenius weight drift  ‖W − W₀‖_F")
    ax.set_xlim(min(ratios) - 0.01, max(ratios) + 0.01)
    panel_label(ax, "D")

    fig.tight_layout()
    save_both(fig, output_dir, "fig7_capacity")

# =============================================================================
# Figure 8 — Sequential coherent perturbation control
# =============================================================================

def fig8_sequential_clamping(R, output_dir):
    print("Figure 8: sequential clamping control...")
    if "sequential_clamping" not in R:
        print("  SKIPPED — no 'sequential_clamping' key in results.pkl")
        return
    sc = R["sequential_clamping"]
    rotations = sc["rotations"]
    alpha_total = sc["alpha"]

    # Per-node duration α_per_node = alpha_total / k
    alpha_per_node = [alpha_total / k for k in rotations]
    persist_pct  = [sc[f"k_{k}"]["persist_pct_node0"] for k in rotations]
    recovery_pct = [sc[f"k_{k}"]["recovery_pct"]      for k in rotations]
    drifts       = [sc[f"k_{k}"]["drift_mean"]        for k in rotations]
    drift_sd     = [sc[f"k_{k}"]["drift_std"]         for k in rotations]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # x positions: use k-index spacing (evenly spaced) rather than log(α),
    # so the four conditions are evenly distributed and uncluttered.
    x = np.arange(len(rotations))
    xticklabels = [f"k = {k}\n(α/node = {int(ap)})"
                   for k, ap in zip(rotations, alpha_per_node)]

    # Panel A: persistence vs recovery across rotation count
    ax = axes[0]
    ax.plot(x, persist_pct, color=C_BURGUNDY, marker="o",
            markersize=8, linewidth=1.8, label="Node-0 persistence")
    ax.plot(x, recovery_pct, color=C_DARKGRAY, marker="s",
            markersize=7, linewidth=1.5, linestyle="--", label="Recovery to ξ⁽¹⁾")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylabel("Percentage of seeds")
    ax.set_ylim(-5, 110)
    ax.legend(loc="center left", frameon=False, fontsize=9)
    panel_label(ax, "A")

    # Panel B: Frobenius weight drift across rotation count
    ax = axes[1]
    ax.errorbar(x, drifts, yerr=drift_sd,
                color=C_MUTEDBLUE, marker="s", markersize=7, linewidth=1.5,
                capsize=4, ecolor=C_DARKGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylabel("Frobenius weight drift  ‖W − W₀‖_F")
    panel_label(ax, "B")

    fig.tight_layout()
    save_both(fig, output_dir, "fig8_sequential_clamping")


# =============================================================================
# Supplementary Figure 1 — Weight-decay robustness
# =============================================================================

def figS1_weight_decay(R, output_dir):
    print("Figure S1: weight-decay robustness...")
    wd  = R["weight_decay"]
    pn  = wd["plasticity_necessity_wd"]
    pt  = wd["phase_transition_wd"]
    ws  = wd["weight_swap_wd"]
    law = wd["scaling_2_over_N_wd"]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))

    # Panel A: plasticity necessity
    ax = axes[0, 0]
    etas = sorted(pn.keys())
    rates = [pn[e]["rate"] for e in etas]
    ax.bar([0, 1], rates, color=[C_DARKGRAY, C_BURGUNDY],
           alpha=0.85, edgecolor="none", width=0.55)
    # Annotate each bar with its value (makes zero-height bars visible)
    for x, r in zip([0, 1], rates):
        ax.text(x, r + 3, f"{r:.0f}%", ha="center", va="bottom",
                fontsize=10, color=C_DARKGRAY)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"η = {e}" for e in etas])
    ax.set_ylabel("Recovery rate (%)")
    ax.set_ylim(0, 115)
    panel_label(ax, "A")

    # Panel B: phase-threshold sweep
    ax = axes[0, 1]
    pt_etas    = [r["eta"]         for r in pt]
    pt_persist = [r["persist_pct"] for r in pt]
    pt_drift   = [r["drift_mean"]  for r in pt]
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
    ax.plot(pt_etas, pt_persist, color=C_BURGUNDY, marker="o",
            markersize=4, linewidth=1.5)
    ax2.plot(pt_etas, pt_drift, color=C_DARKGRAY, marker="s",
             markersize=3.5, linewidth=1.1, linestyle="--")
    ax.set_xlabel("Plasticity rate η")
    ax.set_ylabel("Persistence (%)", color=C_BURGUNDY)
    ax2.set_ylabel("Weight drift", color=C_DARKGRAY)
    ax.tick_params(axis="y", labelcolor=C_BURGUNDY)
    ax2.tick_params(axis="y", labelcolor=C_DARKGRAY)
    ax.set_ylim(-5, 105)
    panel_label(ax, "B")

    # Panel C: weight-swap
    ax = axes[1, 0]
    ws_rates = [ws["rec_with_orig_pct"], ws["rec_with_plastic_pct"]]
    ax.bar([0, 1], ws_rates,
           color=[C_DARKGRAY, C_BURGUNDY], alpha=0.85, edgecolor="none",
           width=0.55)
    for x, r in zip([0, 1], ws_rates):
        ax.text(x, r + 3, f"{r:.0f}%", ha="center", va="bottom",
                fontsize=10, color=C_DARKGRAY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["W^(original)", "W^(plastic)"])
    ax.set_ylabel("Recovery rate (%)")
    ax.set_ylim(0, 115)
    panel_label(ax, "C")

    # Panel D: 1 − 2/N law
    ax = axes[1, 1]
    Ns   = sorted(law.keys())
    obs  = [law[N]["ov_mean"] for N in Ns]
    pred = [law[N]["predicted_1_minus_2_over_N"] for N in Ns]
    ax.plot(Ns, pred, color=C_LIGHTGRAY, linewidth=1.6, linestyle="--",
            label="1 − 2/N (predicted)")
    ax.scatter(Ns, obs, s=60, color=C_BURGUNDY, zorder=3,
               edgecolor="none", label="Observed")
    ax.set_xlabel("Network size N")
    ax.set_ylabel("Post-release overlap")
    ax.set_xlim(min(Ns) - 20, max(Ns) + 20)
    ax.set_ylim(0.94, 1.005)
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "D")

    fig.tight_layout()
    save_both(fig, output_dir, "figS1_weight_decay_robustness")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/results.pkl",
                        help="Path to results.pkl (default: results/results.pkl)")
    parser.add_argument("--output-dir", default="figures",
                        help="Output directory (default: ./figures)")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir   = Path(args.output_dir)

    if not results_path.exists():
        print(f"ERROR: results file not found at {results_path}")
        print("Run reproduce_paper.py first to generate it.")
        sys.exit(1)

    print(f"Loading {results_path}...")
    with open(results_path, "rb") as f:
        R = pickle.load(f)

    if "metadata" in R:
        m = R["metadata"]
        print(f"  Generated: {m.get('timestamp', '?')}")
        print(f"  Python {m.get('python_version', '?')}, "
              f"NumPy {m.get('numpy_version', '?')}, "
              f"runtime {m.get('runtime_hours', '?')} h")

    setup_style()

    print(f"\nWriting figures to {output_dir.resolve()}")
    fig1_phase_threshold(R, output_dir)
    fig2_focal_vs_diffuse(R, output_dir)
    fig3_weight_swap_and_interpolation(R, output_dir)
    fig4_heatmap(R, output_dir)
    fig5_scaling(R, output_dir)
    fig6_perturbation_taxonomy(R, output_dir)
    fig7_capacity(R, output_dir)  
    fig8_sequential_clamping(R, output_dir)
    figS1_weight_decay(R, output_dir)
    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
