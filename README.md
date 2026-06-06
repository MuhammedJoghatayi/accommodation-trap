# The Accommodation Trap

This repository contains the simulation code, saved numerical outputs, and figure-generation scripts for the manuscript:

**The Accommodation Trap: An Attractor Model of Plasticity-Driven Persistence in Delusion-Like Fixation**

The repository is intended to support reproducibility of the manuscript’s main analyses, and figures.

---

## Quick start

```bash
git clone https://github.com/MuhammedJoghatayi/accommodation-trap.git
cd accommodation-trap
pip install -r requirements.txt
python src/reproduce_paper.py
python src/generate_figures.py
```

After running both scripts, the `results/` directory contains the numerical results pickle and the `figures/` directory contains all main and supplementary figures.

---

## Repository structure

```
accommodation-trap/
│
├── src/
│   ├── reproduce_paper.py
│   └── generate_figures.py
│
├── results/
│   └── results.pkl
│
├── figures/
│   ├── fig1_phase_threshold.eps
│   ├── fig1_phase_threshold.pdf
│   ├── fig1_phase_threshold.png
|   |
│   ├── fig2_focal_vs_diffuse.eps
│   ├── fig2_focal_vs_diffuse.pdf
│   ├── fig2_focal_vs_diffuse.png
|   |
│   ├── fig3_weight_swap_and_interpolation.eps
│   ├── fig3_weight_swap_and_interpolation.pdf
│   ├── fig3_weight_swap_and_interpolation.png
|   |
│   ├── fig4_heatmap.eps
│   ├── fig4_heatmap.pdf
│   ├── fig4_heatmap.png
|   |
│   ├── fig5_scaling.eps
│   ├── fig5_scaling.pdf
│   ├── fig5_scaling.png
|   |
│   ├── fig6_perturbation_taxonomy.eps
│   ├── fig6_perturbation_taxonomy.pdf
│   ├── fig6_perturbation_taxonomy.png
|   |
│   ├── fig7_capacity.eps
│   ├── fig7_capacity.pdf
│   ├── fig7_capacity.png
|   |
│   ├── fig8_sequential_clamping.eps
│   ├── fig8_sequential_clamping.pdf
│   ├── fig8_sequential_clamping.pdf
|   |
│   ├── figS1_weight_decay_robustness.eps
│   ├── figS1_weight_decay_robustness.pdf
│   └── figS1_weight_decay_robustness.png
│
├── README.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Dependencies

```
python >= 3.10
numpy >= 1.26
matplotlib >= 3.7
```

Exact pinned versions are in `requirements.txt`. The pipeline has been verified with Python 3.13 and NumPy 2.4.

```bash
pip install -r requirements.txt
```

---

## Running the analyses

### Full pipeline

```bash
python reproduce_paper.py
```

Progress is printed to stdout and mirrored to `results/results.log` for later inspection.

## Main analyses

1. The main pipeline includes:

   1. plasticity necessity analysis;
   2. focal coherent fixation versus diffuse incoherent perturbation;
   3. phase-threshold sweep across plasticity rate;
   4. weight-swap causal control;
   5. structural interpolation between original and accommodated weight matrices;
   6. basin-escape heatmap across accommodation parameters;
   7. distributed accommodation and connectivity-bias analysis;
   8. secondary perturbation taxonomy;
   9. network-size scaling.

   ## Robustness checks

   The robustness pipeline includes:

   1. clamped-node substitution;
   2. weight-decay normalization;
   3. memory-load robustness;
   4. sequential-clamping / sustained-duration control.

## Outputs

Simulation outputs are saved in the `results/` directory. Figures are saved in the `figures/` directory.

Key outputs include:

```bash
results.pkl
metadata.json
```

### Generating figures

```bash
python generate_figures.py
```

Reads `results/results.pkl` and writes all main and supplementary figures to `figures/`, in PDF (vector), eps and PNG formats. 

---

## Reproducibility notes

Every analysis uses deterministic seeds via `np.random.RandomState(seed)`. Bit-for-bit reproducibility across machines requires matching NumPy version (see `requirements.txt`); results have been verified at NumPy 2.4. Asynchronous Hopfield updates select nodes from a per-seed RNG, so identical seeds produce identical update sequences.

---

## Citing

If you use this code or generated outputs, please cite the archived Zenodo release:

```http
[![DOI](https://zenodo.org/badge/1261475707.svg)](https://doi.org/10.5281/zenodo.20573655)
```

A `CITATION.cff` file is included for citation metadata.

---

## License

Code is released under the MIT License.

---

## Contact

For questions about reproducing the results or for the manuscript itself, contact the corresponding author at: muhammedjyi@gmail.com
