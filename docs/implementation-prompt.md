# Implementation Prompt for Claude Sonnet — Foveated Vision Thesis Notebooks

> **Usage:** Provide the entire text below (including this heading) to Claude Sonnet as a
> single task prompt. The model must have access to the files in the working directory
> (`Biological_Vision_Models_Thesis_Report.docx` [primary] and `.pdf`,
> `technical-guide.md`, `experiment-plan-and-ablations.md`, `paper-repositories.md`,
> `trace-based-noise-guide.md`, `vonenet/`).

---

## ROLE AND GOAL

You are a senior research engineer setting up the experimental infrastructure for a
master's thesis on biologically-inspired vision models. Your task is to produce **7
Jupyter notebooks (`.ipynb`)** from scratch, fully executable and **mathematically
grounded**. This is the reference implementation for the architecture proposed in the
thesis (attention-driven foveation + V1 front-end + recurrent ventral stream + IT-feedback
loop; original contribution: trace-based metameric peripheral noise).

## READ FIRST (context files)

Before writing any code, read and internalize the following files:

1. `Biological_Vision_Models_Thesis_Report.docx` — **the current/primary version of the
   thesis** (architecture §8, biological mapping §9, experiment plan §10). This document
   wins in case of conflicts. (The `.pdf` is an older copy; use only for general context.)
2. `technical-guide.md` — Colab setup, notebook organization, and **override strategy**
   (modifying libraries; use subclass / monkey-patch / wrapper).
3. `experiment-plan-and-ablations.md` — components (C1–C5), baselines (B0–B5), ablation
   matrix (A/B/D/E groups), metrics, and MFTMA methodology.
4. `paper-repositories.md` — official repos to be used and papers to be reimplemented.
5. `trace-based-noise-guide.md` — mathematical framework of the **original contribution**
   (trace-based metameric periphery): SNR formalism + Watson (2014) mRGC grounding.
   The math core of Notebook 02 follows this verbatim.
6. `vonenet/vonenet/` — actual implementation of VOneBlock (`modules.py`, `vonenet.py`).
   Write overrides based on this.

> Note: The current thesis (docx) does **not** include a subcortical/EVNets block, but
> **does** include the Watson (2014) [23] mRGC grounding. Follow the docx over the PDF
> in case of conflicts.

## BINDING RULES (do not violate)

1. **Environment is Google Colab.** Every notebook starts with the standard opening cell
   from `technical-guide.md §1.1` (Drive mount, `IN_COLAB` detection, `PROJECT_ROOT`,
   checkpoint directory). Pin dependency versions. **Checkpoint + resume** is mandatory
   for long training runs.
2. **Every deliverable is a `.ipynb`.** Produce real notebook JSON (nbformat v4). Shared
   utilities go in `src/*.py`, but **experiment logic and overrides must be visible inside
   notebook cells** (for reproduction transparency).
3. **Override, don't rewrite.** Repos are cloned to `external/`, added to `sys.path`,
   **never edited**. Behavioral changes are made via subclass/monkey-patch/wrapper (see
   `technical-guide.md §3`). Every monkey-patch must have a rollback path.
4. **Mathematical background is mandatory.** Every notebook contains cells explaining the
   math of its method in **Markdown + LaTeX** *before* code cells (follow the "MATH CORE"
   lists below verbatim). Equations use `$...$` / `$$...$$`; all symbols are defined;
   code implements the written equations exactly and comments reference the equations.
5. **Reproducibility.** Fixed seed (torch/numpy/random + cudnn deterministic),
   `CFG` dict, write results to `results/NN_*.json` and figures to `results/NN_*.png`.
   Use `vone_block.fix_noise(seed=...)` during evaluation since VOneBlock is stochastic.
6. **Stochastic model = EOT required.** For adversarial evaluation (PGD/APGD) with
   stochastic front-end/periphery, compute gradients **EOT-N (N≥10)**; otherwise
   robustness numbers are artificially inflated.
7. **Small scale first.** Every method is validated on CIFAR-10 first, then scaled to
   ImageNet(-100/-1K). Notebooks must support both via a `DATASET` flag.
8. **Honesty.** If something doesn't work or was skipped, state it clearly in the notebook
   (TODO/WARNING cell). Do not fabricate results or pseudo-metrics.
9. **All output in English.** Every produced **code and artifact is in English**: code,
   variable/function names, comments, docstrings, Markdown explanation cells (including
   math), `print`/log messages, figure titles and axis labels, `README.md`. (Only these
   planning documents remain in Turkish; you may reference them but produce notebook
   content in English.)

## NOTEBOOKS TO PRODUCE AND THEIR MATH CORES

Produce in order. Each must be self-contained but may read the previous notebook's output.

### `notebooks/00_setup_and_data.ipynb`
- Content: Drive mount, repo cloning (`paper-repositories.md §3`), data preparation
  (CIFAR-10, ImageNet-100/1K, ImageNet-C/R/Sketch, Stylized-ImageNet), normalization
  (VOneNet: mean=std=0.5), `src/` skeleton.
- **MATH CORE:** data normalization, train/val split; ImageNet-C severity levels and
  **mCE** definition: $\mathrm{CE}_c = \frac{\sum_s E_{s,c}}{\sum_s E_{s,c}^{\text{AlexNet}}}$,
  $\mathrm{mCE}=\frac{1}{15}\sum_c \mathrm{CE}_c$.

### `notebooks/01_baseline_reproduce.ipynb`  (Baselines B0–B3)
- Content: load/train ResNet-50, AlexNet, VOneResNet-50, CORnet-S; verify baseline gain
  on one robustness slice (ImageNet-C or PGD).
- **MATH CORE (VOneBlock — LNP model):**
  - Gabor filter: $g(x,y)=\exp\!\big(-\frac{x'^2}{2\sigma_x^2}-\frac{y'^2}{2\sigma_y^2}\big)\cos(2\pi f x' + \phi)$,
    $x'=x\cos\theta+y\sin\theta$. Simple/complex cell channels.
  - Simple cell: $r_s=[\,k\,(g_{q0}*I)\,]_+$ (half-wave rectification). Complex cell (quadrature):
    $r_c=\frac{k}{\sqrt2}\sqrt{(g_{q0}*I)^2+(g_{q1}*I)^2}$.
  - Neural (Poisson-like) stochasticity: $\tilde r = r + \xi\sqrt{\mathrm{ReLU}(r)+\epsilon}$,
    $\xi\sim\mathcal N(0,1)$ (see `modules.py: noise_f`, neuronal mode). Parameters:
    `k_exc`, `noise_scale`, `noise_level`.
  - PGD attack: $x_{t+1}=\Pi_{\|\delta\|_\infty\le\varepsilon}\big(x_t+\alpha\,\mathrm{sign}\nabla_x \ell\big)$.
  - Note: Liao & Poggio (2016) — ResNet residual blocks are equivalent to **unrolled
    recurrence** (explain as architectural justification).

### `notebooks/02_foveation_rblur_and_periphery.ipynb`  (Ablations A1–A6, E1–E3) — ORIGINAL CORE
- Content: R-Blur foveal transform (reimplement, [9]), three periphery regimes
  (flat blur / i.i.d. noise / **trace-based metameric**), blur-intensity curriculum,
  differentiable foveal warp (FOVEA [3] reference).
- **MANDATORY VISUALIZATION (trace-based noise sanity-check):** Before any training,
  take a few samples from the datasets (≥6 samples from CIFAR-10 and ImageNet) and
  visualize them with trace-based noise applied. For each sample, draw a side-by-side panel:
  `[original] · [R-Blur foveal] · [flat blur periphery] · [i.i.d. noise periphery] · [trace-based metameric periphery]`,
  with fixation point marked and the measured $\mathrm{SNR}(r)$ value printed. Additionally,
  plot the $\mathrm{SNR}(r)$ curve vs eccentricity and the effect of the $\beta$ sweep.
  Save figures to `results/02_periphery_samples.png` and `results/02_snr_profile.png`.
  Purpose: verify visually what noise does to the image before numerical metrics.
- **MATH CORE:**
  - Eccentricity-dependent acuity: distance $r=\|\mathbf p-\mathbf p_0\|$ from fixation
    $\mathbf p_0$; blur scale $\sigma(r)=\sigma_0(1+\kappa r)$ (linear) or cortical
    magnification $M(r)\propto 1/(r+r_0)$. Spatially varying convolution via multi-scale
    Gaussian pyramid.
  - Color saturation reduction: $s(r)=\max(0,\,1-\lambda r)$, applied to HSV/S channel.
  - **Trace-based metameric periphery (original contribution) — SNR-based.** The math of
    this section must be identical to `trace-based-noise-guide.md`:
    - **Watson (2014) pooling scale:** $60\,s(r)\approx 0.53+0.434\,r$, window $w(r)=\kappa\,s(r)$.
    - **Local summary statistic** $S_\Omega$: AdaIN ($\mu,\sigma$) / VGG Gram $G_{ij}^\ell=\sum_k F_{ik}^\ell F_{jk}^\ell$ / Portilla–Simoncelli.
    - **Metamer synthesis:** $\hat I\sim P(\cdot\mid S_{\Omega(r)}(I))$; optimization
      $\min_{\hat I}\sum_{\text{region}}\|S_\Omega(\hat I)-S_\Omega(I)\|^2$ or amortized network.
    - **SNR formalism (core):** $\mathrm{SNR}(r)=\dfrac{P_{\text{signal}}(r)}{P_{\text{noise}}(r)}
      =\mathrm{SNR}_0\big(s_0/s(r)\big)^{\beta}$; $\mathrm{SNR}_{\text{dB}}=10\log_{10}\mathrm{SNR}$.
    - **SNR knob:** content–texture interpolation $\hat I_\alpha=\alpha(r)\,I+(1-\alpha(r))\,T(S_\Omega(I))$,
      $\alpha(r)=\dfrac{\sqrt{\mathrm{SNR}(r)/\rho}}{1+\sqrt{\mathrm{SNR}(r)/\rho}}$.
  - **Unify three regimes on the SNR axis:** flat blur (low-pass, no added noise) /
    i.i.d. noise ($\eta\sim\mathcal N$, signal-independent — like VOneNet Poisson) /
    trace-based (signal-conditioned, texture-matched). A2–A4 compared at **equal average
    SNR budget**; difference comes from noise *structure*. **Measure and report the realized
    $\mathrm{SNR}(r)$ curve** (main figure: metric vs SNR).
  - Composite image: sharp fovea $\odot$ mask + metameric periphery $\odot$ (1−mask).

### `notebooks/03_v1_block.ipynb`  (Ablations B7–B9, B12–B13)
- Content: V1 front-end ablation (none / VOneBlock noise-off / Poisson-on) via `noise_f`
  override; ventral back-end swap (ResNet-50 ↔ CORnet-S).
- **MATH CORE:**
  - VOneBlock LNP (reference above) + effect of noise on/off ablation on gradient flow.
  - **Brain-Score predictivity:** model activation → neural response via PLS/linear regression;
    explained variance $R^2$, score normalized to noise ceiling.
  - CORnet-S recurrence: $h_t = f(W_{ff}x + W_{rec}h_{t-1})$; unrolling and IT dynamics.

### `notebooks/04_it_feedback_multiglance.ipynb`  (Ablations D1–D4)
- Content: attention/saliency → fixation selection (top-k/NMS) → multi-glance loop →
  confidence/uncertainty halting; differentiable (soft) vs REINFORCE (hard) variants
  (GFNet [15], RAM [16], NeVA [17] references). `FoveatedModel` wrapper (`technical-guide.md §3.3`).
- **MATH CORE:**
  - Saliency/attention rollout: self-attention layers $\tilde A=\prod_\ell (A_\ell+I)$;
    top-k / NMS for fixation.
  - Uncertainty: softmax $p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$; **entropy** $H(p)=-\sum_i p_i\log p_i$;
    confidence $=\max_i p_i$. Halting condition: stop when $\max_i p_i \ge \tau$.
  - **REINFORCE (hard attention):** policy $\pi_\phi(l\mid s)$; objective $J=\mathbb E[R]$;
    gradient $\nabla_\phi J=\mathbb E\big[R\,\nabla_\phi\log\pi_\phi(l\mid s)\big]$; baseline
    $b$ for variance: $(R-b)\nabla_\phi\log\pi_\phi$. Reward: correct classification.
  - **Differentiable (soft) path:** task loss backpropagated through all glances; soft
    (softmax-weighted) sampling / Gumbel-softmax for fixation.
  - **Counterfactual loss:** additional term rewarding a fixation location if it increases
    the correct-class probability.
  - Adaptive computation / evidence accumulation (ACC analogy): expected gain threshold.

### `notebooks/05_mftma_certification.ipynb`  (All conditions × MFTMA + metamer)
- Content: extract class-manifolds from selected layers (V1 output, ventral middle, IT
  readout) for each ablation condition; certify with `neural_manifolds_replicaMFT`
  ([20]/[21]); metamer recognizability ([12]); compare with adversarial-manifolds ([22]).
- **MATH CORE (replica mean-field manifold analysis):**
  - Linear separability of $P$ manifolds in $N$ dimensions; load $\alpha=P/N$;
    **manifold capacity** $\alpha_c$ (critical load for separability under random dichotomies).
  - **Manifold radius** $R_M$ and **manifold dimension** $D_M$ (from anchor points and
    Gaussian statistics); **center correlation**. Relation: low $R_M, D_M$ and low center
    correlation $\Rightarrow$ high $\alpha_c$.
  - Hypothesis testing: as foveation + V1 noise + trace-based periphery + IT feedback
    are added, does $\alpha_c\uparrow$, $R_M,D_M\downarrow$? (behavioral + geometric dual signature).

### `notebooks/06_full_model_and_ablations.ipynb`
- Content: full closed-loop model (C1=R-Blur, C2=trace-based, C3=VOneBlock Poisson,
  C4=ResNet-50, C5=multi-glance+halting); run the full ablation matrix;
  **leave-one-out** summary table (`experiment-plan-and-ablations.md §8`); result figures/tables.
- **MATH CORE:** marginal contribution = full score − (component removed) score;
  efficiency curve (accuracy vs average glances / GFLOP); adversarial gradient via
  EOT-N averaging $\bar g=\frac1N\sum_{i}\nabla_x \ell(f(T_i(x)),y)$.

## WORKFLOW

- Produce notebooks **in numbered order**; before producing each one, inspect the
  relevant repo/code sections (do not assume — read `vonenet` code and any cloned repos).
- Begin each notebook with a short summary cell: "Purpose / Math in this notebook /
  Ablation IDs / Inputs-Outputs".
- Make heavy training steps runnable in 1–2 iterations with `CFG['smoke_test']=True`
  (for fast Colab validation); full training controlled by a separate flag.
- Move shared code to `src/`: `overrides.py`, `foveation.py`, `it_feedback.py`,
  `mftma.py`, `eval_harness.py`.
- At the end, include a short `notebooks/README.md` documenting execution order and
  outputs produced by each notebook.

## ACCEPTANCE CRITERIA

- [ ] 7 notebooks + `src/` modules + `notebooks/README.md` produced.
- [ ] **All notebook content in English** (code, comments, docstrings, Markdown/math
      cells, print/log, figure labels, README).
- [ ] Every notebook includes Colab opening cell, `CFG`, seed, checkpoint/resume.
- [ ] Every notebook covers the "MATH CORE" items listed above with LaTeX Markdown
      cells and code that implements the written equations.
- [ ] Notebook 02 produces the **trace-based noise visualization** before training
      (sample panels + $\mathrm{SNR}(r)$ curve, saved to `results/`).
- [ ] No upstream repo was edited; all changes are overrides.
- [ ] Adversarial evaluations use EOT.
- [ ] All notebooks run end-to-end without errors in `smoke_test` mode.

**First step:** Read the context files, produce the `src/` skeleton and
`notebooks/00_setup_and_data.ipynb`, then present a summary for confirmation.
