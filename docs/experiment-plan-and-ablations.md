# Experiment Plan and Ablation Matrix

> **Thesis:** Attention-driven foveation + V1 front-end + recurrent ventral stream +
> IT-feedback loop; **original contribution:** trace-based metameric peripheral noise.
> **Certification:** MFTMA (manifold capacity/radius/dimension).
> This document expands Chapters 8, 9, and 10 of the thesis report (proposed architecture,
> biological mapping, experimental plan) into an actionable **ablation matrix + methodology**.
>
> For implementation rules (Colab, `.ipynb`, override): `technical-guide.md`.
> For repository links: `paper-repositories.md`.

---

## 1. Model Components (ablation axes)

The proposed architecture is decomposed into 5 independently toggleable components. Each
component is an **ablation axis**; each axis is controlled by a single override flag.

| # | Component | Ablation values | Override target |
|---|-----------|-----------------|-----------------|
| C1 | **Foveation (spatial acuity)** | off / R-Blur foveal warp | `Foveate` wrapper prepended to input |
| C2 | **Periphery regime** | flat blur / i.i.d. noise / **trace-based metameric** (original, Watson-based SNR) | `Periphery` strategy parameter |
| C3 | **V1 front-end (Gabor + noise)** | none / VOneBlock (noise off) / VOneBlock (Poisson on) | `VOneBlock.noise_f` override |
| C4 | **Ventral back-end** | ResNet-50 (main) / CORnet-S (recurrent reference) | `VOneNet(model_arch=...)` |
| C5 | **IT feedback + fixation** | single-glance / multi-glance + confidence-halting | `FoveatedModel` loop + `it_head` |

**Full model:** C1=R-Blur, C2=trace-based, C3=VOneBlock(Poisson on),
C4=ResNet-50, C5=multi-glance+halting.

---

## 2. Baselines (reference points)

| Baseline | Purpose | Source/override |
|----------|---------|-----------------|
| **B0 · ResNet-50** | Main feedforward backbone, scale reference | torchvision |
| **B1 · AlexNet** | Classic weak baseline (VOneNet comparison) | torchvision |
| **B2 · VOneResNet-50** | V1 front-end contribution anchor | `vonenet` [1] |
| **B3 · CORnet-S** | Recurrent ventral stream reference (Brain-Score) | `CORnet` [6] |
| **B4 · R-Blur ResNet-50** | Foveal-training robustness reference | reimpl [9] |
| **B5 · LCL-V2Net** | Self-supervised V1/V2 front-end reference | `LCL-V2` [11] |

> B0–B3 are **mandatory reproductions** (Phase 1). B4–B5 are added in the relevant ablation phase.

---

## 3. Ablation Matrix (experiment table)

Each row is an experimental condition. The `→` column shows **which scientific question**
that ablation isolates. Priority: **P1** (core, mandatory) > **P2** (important) > **P3** (if time permits).

### 3.1 Foveation & Periphery (original contribution axis) — Notebook 02

| ID | C1 Foveation | C2 Periphery | → Isolated question | Priority |
|----|-------------|-------------|---------------------|----------|
| A1 | off | — | Baseline without foveation (full-res) | P1 |
| A2 | R-Blur | flat blur | Classic foveal blur contribution | P1 |
| A3 | R-Blur | i.i.d. noise (signal-independent) | Unstructured noise vs blur | P2 |
| A4 | R-Blur | **trace-based metameric** | **Net effect of original contribution** | **P1** |
| A5 | R-Blur | trace-based (blur/fixation curriculum) | Curriculum contribution | P2 |
| A6 | R-Blur | trace-based, **SNR(r) profile sweep** | SNR roll-off and biological grounding effect | P1 |

**A2–A4 comparison rule:** all three periphery regimes are matched at **equal average SNR
budget**; the difference then comes from the *structure* of the noise (signal-conditioned,
texture-matched) rather than the SNR level. Formulation: `trace-based-noise-guide.md §3–4`.

**A6 (SNR sweep):** in the trace-based regime, the pooling scale is linked to the
**Watson (2014) mRGC** range: $60\,s(r)\approx 0.53+0.434\,r$; SNR roll-off
$\mathrm{SNR}(r)=\mathrm{SNR}_0\,(s_0/s(r))^{\beta}$. Swept: $\beta\in\{1,2,3\}$,
$\mathrm{SNR}_0$ levels; and **"Watson-based $\mathrm{SNR}(r)$" vs "flat SNR"
vs "arbitrary blur schedule"**. Goal: accuracy–robustness sweet spot + biological grounding contribution.

**Measured:** clean accuracy, ImageNet-C mCE, PGD/APGD (EOT), shape bias, GFLOP/latency,
**realized $\mathrm{SNR}(r)$ curve** (eccentricity-binned). Main figure: metrics plotted
**as a function of SNR**.
**Hypothesis:** A4 > A3 ≳ A2 > A1 in robustness (trace-based leads even at equal SNR);
Watson-based $\mathrm{SNR}(r)$ (A6) closest to human peripheral texture representation —
verified by metamer recognizability (low human-discriminability) + MFTMA capacity.

### 3.2 V1 front-end — Notebook 03

| ID | C3 V1 | → Isolated question | Priority |
|----|-------|---------------------|----------|
| B7 | none | Baseline without front-end | P1 |
| B8 | VOneBlock (noise off) | Gabor bank contribution alone | P1 |
| B9 | VOneBlock (Poisson on) | Stochastic noise robustness contribution | P1 |

**Measured:** V1/V2 Brain-Score predictivity, adversarial robustness (EOT), shape bias.
**Note:** EOT-N required for B9; prevent noise gradient masking.

### 3.3 Ventral back-end & Recurrence — Notebook 03/04

| ID | C4 Back-end | → Isolated question | Priority |
|----|------------|---------------------|----------|
| B12 | ResNet-50 | Feedforward baseline (residual = unrolled recurrence, Liao & Poggio 2016) | P1 |
| B13 | CORnet-S | Explicit recurrence Brain-Score/robustness contribution | P2 |

**Measured:** V4/IT Brain-Score, ImageNet accuracy, parameter/FLOP efficiency.

### 3.4 IT Feedback & Multi-glance — Notebook 04

| ID | C5 loop | Halting | Training signal | → Isolated question | Priority |
|----|---------|---------|----------------|---------------------|----------|
| D1 | single-glance | — | task loss | Feedback-free baseline | P1 |
| D2 | multi-glance | confidence-threshold | differentiable (soft) | Fixation repetition contribution | P1 |
| D3 | multi-glance | confidence-threshold | REINFORCE (hard attention) | RL vs differentiable comparison | P2 |
| D4 | multi-glance | entropy/uncertainty-threshold | soft + counterfactual loss | Uncertainty-driven fixation contribution | P3 |

**Measured:** accuracy vs average glance count (efficiency curve), adaptive-compute
gain, accuracy on small/occluded objects, latency.
**References:** GFNet [15] (confidence halting), RAM [16] (REINFORCE), NeVA [17] (foveated attention).

### 3.5 Training regime ablation — all notebooks

| ID | Regime | → Isolated question | Priority |
|----|--------|---------------------|----------|
| E1 | Fixed blur intensity | Baseline | P2 |
| E2 | Blur/fixation **curriculum** | Curriculum biology-encoding contribution | P1 |
| E3 | Curriculum + adversarial-free | Is robustness possible without AT? | P1 |

---

## 4. Datasets

| Dataset | Usage | Phase |
|---------|-------|-------|
| **CIFAR-10** | Fast iteration / prototyping (all ablations start here) | 2–4 |
| **ImageNet-1K** | Main clean accuracy and scale | 1–5 |
| **ImageNet-C** | Common corruption robustness (mCE) | 1–5 |
| **ImageNet-R / Sketch** | Domain shift, OOD | 3–5 |
| **Stylized-ImageNet (Geirhos)** | Shape bias / cue-conflict | 5 |
| **ImageNet-100** | FoveaTer-style efficiency comparison | 2, 4 |
| **Brain-Score neural sets** | V1/V2/V4/IT predictivity | 3–5 |

> **Methodological rule:** every ablation is first validated quickly on CIFAR-10, then
> scaled to ImageNet (thesis report §11 note). Original components such as IT-feedback and
> trace-based periphery are prioritized.

---

## 5. Metrics

| Category | Metric | Tool |
|----------|--------|------|
| **Accuracy** | top-1/top-5 clean accuracy | standard |
| **Corruption** | mCE (ImageNet-C), ImageNet-R/Sketch acc | `eval_harness` |
| **Adversarial** | PGD, APGD robust acc — **EOT-N (N≥10)** for stochastic models | foolbox/robustbench |
| **Brain alignment** | V1/V2/V4/IT predictivity | Brain-Score [7] |
| **Human alignment** | shape bias, cue-conflict accuracy | Geirhos dataset |
| **Perceptual gap** | metamer recognizability | model_metamers_pytorch [12] |
| **Geometry (certification)** | manifold capacity, radius, dimension, center-corr | MFTMA [20/21] |
| **Peripheral fidelity** | realized **$\mathrm{SNR}(r)$ curve** (eccentricity-binned), Watson fit error | `foveation.py` measurement |
| **Efficiency** | GFLOP, latency, memory, average fixation count | profiler |

> **SNR main figure:** accuracy / mCE / adversarial acc / manifold capacity, each plotted
> **as a function of the $\mathrm{SNR}(r)$ profile** (three periphery regimes on one axis).
> Formulation and measurement: `trace-based-noise-guide.md`.

---

## 6. MFTMA Certification Methodology

**In addition** to accuracy and Brain-Score, we measure the representational geometry of
each ablation with MFTMA and report a geometric signature (thesis report §10.1).

**Procedure:**
1. For each condition, class-per-activation manifolds are extracted from selected layers
   (V1 output, ventral middle layer, IT readout) — ~50 samples per class.
2. **Capacity (αc), radius (Rm), dimension (Dm), center-correlation** are computed with
   `neural_manifolds_replicaMFT`.
3. Compared across layers and ablation conditions.

**Directly testable hypotheses (from thesis):**
- As foveation + V1 noise + trace-based periphery + IT feedback are added, do object
  manifolds become **more separable**? → **higher capacity, lower radius/dimension** expected.
- Does the capacity gain from stochastic bio-inspired front-ends explain the role of noise
  in robust perception (Dapello, Chung et al. [22])? → compare with adversarial-manifolds findings.

Each ablation is thus reported with **two signatures**: behavioral (accuracy/robustness) +
geometric (capacity).

---

## 7. Experiment → Notebook → Phase Mapping

| Notebook | Ablations covered | Thesis phase (report §11) |
|----------|-------------------|---------------------------|
| `00_setup_and_data` | — (data + repo setup) | Phase 0 (Jul 13–19) |
| `01_baseline_reproduce` | B0–B3 | Phase 1 (Jul 20 – Aug 2) |
| `02_foveation_rblur_and_periphery` | A1–A6, E1–E3 | Phase 2 (Aug 3–16) |
| `03_v1_block` | B7–B9, B12, B13 | Phase 3 (Aug 17–30) |
| `04_it_feedback_multiglance` | D1–D4 | Phase 4 (Aug 31 – Sep 13) |
| `05_mftma_certification` | all conditions × MFTMA + metamer | Phase 5 (Sep 14–23) |
| `06_full_model_and_ablations` | Full model + leave-one-out summary | Phase 6 (Sep 24–28) |

---

## 8. Leave-One-Out Summary Experiment (final)

By **removing each component (C1…C5) one at a time** from the full model, the marginal
contribution is measured:

| Condition | Removed component | Expected drop |
|-----------|-------------------|---------------|
| Full | — | reference (highest) |
| −C1 | foveation | efficiency ↓, small object acc ↓ |
| −C2 | trace-based → flat blur (equal SNR) | human alignment ↓, metamer similarity ↓ |
| −C3 | V1 noise | adversarial robustness ↓ (EOT) |
| −C4 | replace ResNet with CORnet-S | Brain-Score ↑?, FLOP change |
| −C5 | IT feedback | adaptive-compute gain ↓, hard example acc ↓ |

This table provides single-glance evidence for the thesis's main contribution claim
(synergistic contribution of each component).

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Colab session drop / long ImageNet training | Checkpoint+resume; CIFAR-10 first; subset (ImageNet-100) |
| Spurious robustness with stochastic front-end | EOT-N mandatory in all adversarial measurements |
| R-Blur/FoveaTer has no official repo | Reimplement from paper (see `paper-repositories.md` §2) |
| Trace-based periphery synthesis cost | NeuroFovea/SideEye-style amortized small generative network [19] |
| Upstream API change | No forking; override only (see `technical-guide.md` §3) |
| Time pressure (plan starts mid-July) | P1 ablations first; P3 only if time permits |
