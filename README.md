# Foveated VOneNet — Biologically Inspired Vision Models

A thesis research project implementing an attention-driven foveation + V1 front-end + recurrent ventral stream + IT-feedback architecture, with **trace-based metameric peripheral noise** as the original contribution. Robustness is certified geometrically via MFTMA (manifold capacity/radius/dimension).

---

## Overview

The model is composed of five independently toggleable components:

| Component | Description |
|-----------|-------------|
| **C1 Foveation** | R-Blur spatially-varying foveal warp |
| **C2 Periphery** | Trace-based metameric noise (Watson 2014 mRGC-grounded SNR) |
| **C3 V1 front-end** | VOneBlock with optional Poisson noise (VOneNet) |
| **C4 Ventral back-end** | ResNet-50 (primary) / CORnet-S (recurrent reference) |
| **C5 IT feedback** | Multi-glance loop with confidence/uncertainty halting |

**Full model:** all five components active simultaneously.

---

## Project Structure

```
foveated_vonenet/
├── notebooks/               # Numbered experiment notebooks (run in order)
│   ├── 00_setup_and_data.ipynb
│   ├── 01_baseline_reproduce.ipynb
│   ├── 02_foveation_rblur_and_periphery.ipynb
│   ├── 03_v1_block.ipynb
│   ├── 04_it_feedback_multiglance.ipynb
│   ├── 05_mftma_certification.ipynb
│   └── 06_full_model_and_ablations.ipynb
├── src/                     # Shared Python modules
│   ├── overrides.py         # Subclass/monkey-patch overrides for ablations
│   ├── foveation.py         # R-Blur, trace-based periphery transforms
│   ├── it_feedback.py       # Confidence/uncertainty fixation + halting
│   ├── mftma.py             # Manifold capacity certification wrapper
│   └── eval_harness.py      # ImageNet-C, PGD/APGD, Brain-Score bridges
├── external/                # Cloned upstream repos (READ-ONLY)
│   ├── vonenet/             # [1] Dapello et al. 2020
│   ├── CORnet/              # [6] Kubilius et al. 2019
│   └── fovea/               # [3] Thavamani et al. 2021
├── data/                    # Datasets (not tracked by git)
│   ├── cifar10/
│   └── cifar10c/
├── checkpoints/             # Model weights (not tracked by git)
├── results/                 # Metrics (JSON) and figures (PNG)
├── docs/                    # Project documentation (English)
│   ├── experiment-plan-and-ablations.md
│   ├── technical-guide.md
│   ├── trace-based-noise-guide.md
│   ├── paper-repositories.md
│   └── implementation-prompt.md
└── literature/              # PDF references
```

---

## Getting Started

### Environment

Designed for **Google Colab** (A100/L4 GPU). For local execution, set `PROJECT_ROOT` to
the repository path. Every notebook auto-detects the environment with:

```python
IN_COLAB = 'google.colab' in sys.modules
```

### Dependencies

```bash
pip install torch==2.* torchvision timm==1.* foolbox==3.* robustbench \
    scipy scikit-learn matplotlib tqdm
```

### Running notebooks

Execute notebooks in order:

| Step | Notebook | Purpose |
|------|----------|---------|
| 0 | `00_setup_and_data` | Environment setup, repo cloning, dataset preparation |
| 1 | `01_baseline_reproduce` | Reproduce B0–B3 baselines (ResNet-50, AlexNet, VOneResNet-50, CORnet-S) |
| 2 | `02_foveation_rblur_and_periphery` | Foveation & periphery ablations (A1–A6, E1–E3) — **original contribution** |
| 3 | `03_v1_block` | V1 front-end ablations (B7–B9, B12–B13) |
| 4 | `04_it_feedback_multiglance` | IT feedback & multi-glance loop (D1–D4) |
| 5 | `05_mftma_certification` | Geometric certification (manifold capacity/radius/dimension) |
| 6 | `06_full_model_and_ablations` | Full model + leave-one-out ablation summary |

Use `CFG['smoke_test'] = True` in any notebook for a quick 1–2 iteration sanity check.

---

## Documentation

| File | Description |
|------|-------------|
| [docs/experiment-plan-and-ablations.md](docs/experiment-plan-and-ablations.md) | Full ablation matrix, baselines, metrics, and MFTMA methodology |
| [docs/technical-guide.md](docs/technical-guide.md) | Colab setup, notebook organization, override strategy |
| [docs/trace-based-noise-guide.md](docs/trace-based-noise-guide.md) | Mathematical framework for the original contribution (SNR + Watson mRGC) |
| [docs/paper-repositories.md](docs/paper-repositories.md) | Open-source repos for all cited papers |
| [docs/implementation-prompt.md](docs/implementation-prompt.md) | Full prompt for AI-assisted notebook generation |

---

## Key References

| # | Paper | Repo |
|---|-------|------|
| [1] | Dapello et al. 2020 — VOneNet, NeurIPS | [dicarlolab/vonenet](https://github.com/dicarlolab/vonenet) |
| [3] | Thavamani et al. 2021 — FOVEA, ICCV | [tchittesh/fovea](https://github.com/tchittesh/fovea) |
| [6] | Kubilius et al. 2019 — CORnet-S, NeurIPS | [dicarlolab/CORnet](https://github.com/dicarlolab/CORnet) |
| [9] | Shah & Raj 2023 — R-Blur, NeurIPS | arXiv:2308.00854 (reimplemented) |
| [15] | Wang et al. 2020 — GFNet, NeurIPS | [blackfeather-wang/GFNet-Pytorch](https://github.com/blackfeather-wang/GFNet-Pytorch) |
| [20/21] | Chung/Cohen et al. — MFTMA | [schung039/neural_manifolds_replicaMFT](https://github.com/schung039/neural_manifolds_replicaMFT) |
| [23] | Watson 2014 — mRGC density, J. Vision 14(7):15 | DOI:10.1167/14.7.15 (formula, no code) |

See [docs/paper-repositories.md](docs/paper-repositories.md) for the full list.

---

## Important Notes

- **No upstream repos are edited.** All behavioral changes are applied via subclassing,
  monkey-patching, or wrapping (see [docs/technical-guide.md](docs/technical-guide.md) §3).
- **Stochastic models require EOT.** All adversarial evaluations use EOT-N (N≥10) to
  avoid artificially inflated robustness numbers.
- Datasets and checkpoints are **not tracked by git** (see `.gitignore`).
- Results are written to `results/NN_*.json` and `results/NN_*.png`.

---

## Thesis

*Biologically Inspired Vision Models: Attention-Driven Foveation, V1 Front-End, Recurrent
Ventral Stream, and IT-Feedback Loop* — Master's Thesis, Boğaziçi University.
