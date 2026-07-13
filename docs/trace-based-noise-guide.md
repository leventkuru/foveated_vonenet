# Trace-Based Metameric Peripheral Noise — Structure and SNR Guide

> **Link to thesis:** This guide establishes the mathematical framework for the
> trace-based metameric peripheral noise mechanism — the thesis's **original contribution**
> (docx §7 item 4, §8.3, §8.4). The central idea: control peripheral degradation not
> through an arbitrary "blur intensity" parameter, but through **signal-to-noise ratio
> (SNR)**, and ground that SNR profile in **Watson (2014)** human mRGC anatomy.
>
> Parallel source: `Biological_Vision_Models_Thesis_Report.docx` (current version; author:
> Resul Levent Kuru). See `experiment-plan-and-ablations.md` (C2 axis, A1–A5 + SNR sweep)
> and `technical-guide.md §3` (override). Reference: Watson, A. B. (2014), *J. Vision* 14(7):15 [23].

---

## 1. Concept: "trace", "metameric", and why noise?

Human peripheral vision does not erase the signal; it preserves a **statistical trace**
of it. Two key points:

- **Metamer:** In human peripheral vision, two different images that share the same local
  summary statistics (texture statistics) are **indistinguishable** — these are called
  *metamers*. The periphery thus preserves the local statistics, not a full copy, of the
  signal.
- **Trace-based noise:** We replace the periphery not with flat blur (which only removes
  high frequencies) or unstructured i.i.d. noise (which ignores statistics), but with a
  noise sample **matched to the local summary statistics of the underlying signal**
  (signal-conditioned). This sample preserves the statistical trace while randomizing
  the phase and fine structure.

The difference between these three regimes can be summarized in one quantity: **the ratio
of the power of the preserved signal to the power of the added/randomized component** —
i.e., **SNR**. The core of this guide is the formulation that links SNR to eccentricity
and to the Watson mRGC scale.

---

## 2. Watson (2014) and the biological basis of the pooling scale

Eccentricity from fixation $\mathbf p_0$: $r=\lVert \mathbf p-\mathbf p_0\rVert$ (degrees).
Watson (2014) derives, from Curcio & Allen (1990) ganglion density measurements, the
decline of midget RGC receptive-field density with eccentricity:

$$
d_{gf}(r) = d_{gf}(0)\,\Big[a\,\big(1+\tfrac{r}{r_2}\big)^{-2} + (1-a)\,e^{-r/r_e}\Big]
$$

and from this shows that the **mean mRGC spacing** grows approximately linearly with
eccentricity (thesis docx §8.3):

$$
\boxed{\,60\,s(r) \approx 0.53 + 0.434\,r\,}\qquad\Longrightarrow\qquad
s(r) = \frac{0.53 + 0.434\,r}{60}\ \text{[degrees]}
$$

where $s(r)$ is the mean mRGC spacing; the **pooling window radius** is chosen proportional
to this:

$$
w(r) = \kappa\, s(r),\qquad \kappa\gtrsim 1\ \text{(pooling factor, hyperparameter)}.
$$

- $r\to 0$: $s(0)=0.53/60\approx 0.0088^\circ$ (at fovea ≈ 0.5 arcmin, near cone spacing) →
  window ≈ 1 pixel, signal is **preserved exactly**.
- As $r$ grows, $w(r)$ grows linearly → wider window → more statistical summarization.

> Key point: the pooling scale comes from **measured human anatomy**, not from an arbitrary
> blur-σ schedule. This places the mechanism on a quantitative biological foundation
> (thesis docx §7, §8.3).

---

## 3. SNR formalism (core of this guide)

### 3.1 Writing peripheral representation as signal + conditional noise

Let $S_{\Omega}(\cdot)$ be the local summary statistic operator over pooling window $\Omega(r)$
(concretized in Section 5). The trace-based peripheral sample $\hat I$ preserves the
signal's statistics but randomizes the rest:

$$
\hat I \sim P\big(\,\cdot \mid S_{\Omega(r)}(I)\,\big).
$$

Write this as an additive decomposition:

$$
\hat I(\mathbf p) = \underbrace{\bar I_{\Omega(r)}(\mathbf p)}_{\text{preserved signal (trace)}}
\;+\; \underbrace{n(\mathbf p)}_{\text{signal-conditioned metameric noise}},
$$

where $\bar I_{\Omega}$ is the low-variance component within the window (preserved structure)
and $n$ is the residual that matches local texture statistics but has randomized phase.

### 3.2 Eccentricity-dependent local SNR

We define the local SNR as the ratio of preserved signal power to metameric noise power:

$$
\mathrm{SNR}(r) \;=\; \frac{P_{\text{signal}}(r)}{P_{\text{noise}}(r)}
\;=\; \frac{\mathrm{Var}\big[\bar I_{\Omega(r)}\big]}{\mathbb E\big[\lVert I-\hat I\rVert^2_{\Omega(r)}\big]},
\qquad
\mathrm{SNR}_{\text{dB}}(r) = 10\log_{10}\mathrm{SNR}(r).
$$

**Link to Watson (fundamental relation).** The pooling **area** $A(r)\propto w(r)^2\propto s(r)^2$
grows with eccentricity. When summarizing over $\sim A(r)$ samples within the window, fine-structure
fidelity decreases inversely with area. This gives the **roll-off law** for SNR with eccentricity:

$$
\boxed{\;\mathrm{SNR}(r) \;=\; \mathrm{SNR}_0\left(\frac{s(0)}{s(r)}\right)^{\beta}
\;=\; \mathrm{SNR}_0\left(\frac{0.53}{0.53+0.434\,r}\right)^{\beta}\;}
$$

- $\beta=2$: area-based summarization (default, defensible choice).
- $\beta$ is an **ablation knob**: controls the steepness of the peripheral fidelity roll-off.
- $\mathrm{SNR}_0$: foveal (reference) fidelity; typically very high (sharp fovea).

As $r\to 0$, $\mathrm{SNR}\to\mathrm{SNR}_0$ (signal preserved); monotonically decreasing as $r$ grows.

### 3.3 Mapping SNR to a synthesis parameter (actionable knob)

In practice, SNR is controlled via an **interpolation coefficient** $\alpha(r)\in[0,1]$
between content (signal) and texture-sample (noise) (NeuroFovea/AdaIN style):

$$
\hat I_{\alpha}(\mathbf p) = \alpha(r)\,I(\mathbf p) + \big(1-\alpha(r)\big)\,T\big(S_{\Omega(r)}(I)\big),
$$

where $T(\cdot)$ is the texture synthesizer matched to the statistics (Section 5). Under a
small model, the relationship between $\alpha$ and SNR is approximately
$\mathrm{SNR}(\alpha)\approx \dfrac{\alpha^2}{(1-\alpha)^2}\rho$
($\rho$: content/texture power ratio). So the target $\mathrm{SNR}(r)$ profile back-solves to $\alpha(r)$:

$$
\alpha(r) = \frac{\sqrt{\mathrm{SNR}(r)/\rho}}{1+\sqrt{\mathrm{SNR}(r)/\rho}}.
$$

Result: the **Watson $s(r)$ → target $\mathrm{SNR}(r)$ → interpolation $\alpha(r)$** chain
drives peripheral degradation entirely through SNR and with a biological grounding.

---

## 4. Unified view of three periphery regimes on the SNR axis

The thesis's A1–A5 ablation (see `experiment-plan-and-ablations.md §3.1`) is essentially
**three different noise structures on the same SNR axis**:

| Regime | Mathematical structure | SNR behavior | Trace preserved? |
|--------|----------------------|--------------|------------------|
| **Flat blur (R-Blur)** | Low-pass: $\hat I = I * G_{\sigma(r)}$, no added noise | Signal *attenuated*, no separate noise term; high-freq SNR $\to 0$ | Partially (low frequency only) |
| **i.i.d. noise** | $\hat I = I + \eta,\ \eta\sim\mathcal N(0,\sigma^2)$, **signal-independent** | SNR tunable but noise is not conditioned on texture | **No** (statistics not matched) |
| **Trace-based metameric** | $\hat I \sim P(\cdot\mid S_{\Omega(r)}(I))$, $\mathrm{SNR}(r)$ from Watson $s(r)$ | $\mathrm{SNR}(r)=\mathrm{SNR}_0\,(s_0/s(r))^\beta$, **conditional** | **Yes** (local texture statistics) |

This table makes the three regimes **comparable**: by matching all three at equal average
SNR budget, any difference is isolated to come from the **structure** of the noise
(conditioned vs not, texture-matched vs not). The thesis claim: even at equal SNR,
trace-based leads in human alignment and robustness.

---

## 5. Synthesis algorithm (local statistics → metameric sample)

**Summary statistic $S_\Omega$ options (increasing fidelity/cost):**
1. **AdaIN (mean/std):** $S=\{\mu_\Omega, \sigma_\Omega\}$; $\mathrm{AdaIN}(x,y)=\sigma(y)\frac{x-\mu(x)}{\sigma(x)}+\mu(y)$. Cheapest.
2. **VGG Gram matrix:** $G^{\ell}_{ij}=\sum_k F^{\ell}_{ik}F^{\ell}_{jk}$ (Gatys texture code).
3. **Portilla–Simoncelli:** steerable pyramid marginal + cross-correlation statistics. Most biological.

**Synthesis (two paths):**
- **Optimization-based (reference, slow):** $\displaystyle \min_{\hat I}\sum_{\text{region}} \big\lVert S_\Omega(\hat I)-S_\Omega(I)\big\rVert^2$ (Portilla–Simoncelli / Gatys).
- **Amortized (production, fast):** NeuroFovea/SideEye-style small feed-forward generative
  network ($T$). **Required:** synthesis must be cheap; otherwise it eats the efficiency
  gain of foveation (thesis docx §8.3).

**Pipeline (summary):**
```
1. r(p)  = eccentricity map (from fixation)
2. w(p)  = κ·s(r) = κ·(0.53 + 0.434·r)/60           # Watson pooling scale
3. S     = eccentricity-varying windowed S_Ω(I)       # local texture statistics
4. T     = metameric sample matched to S (AdaIN/Gram/PS)
5. α(r)  = from target SNR(r) profile (Section 3.3)
6. Îper  = α(r)·I + (1-α(r))·T
7. comp  = m_fovea·I_sharp + (1-m_fovea)·Îper        # soft-transition composite
```

---

## 6. Link between SNR and thesis claims (why this formulation is valuable)

- **Robustness:** Low peripheral SNR "drowns" adversarial perturbation energy in the
  periphery via metameric noise → robustness ↑. But too low SNR hurts accuracy →
  there is a **SNR–accuracy–robustness sweet spot**. $\beta$ and $\mathrm{SNR}_0$ sweeps
  locate this point.
- **Human alignment:** Matching the $\mathrm{SNR}(r)$ profile to Watson mRGC aims to make
  peripheral outputs **metameric** (indistinguishable) to humans → low human-discriminability
  expected in the metamer recognizability test (Feather [12]).
- **MFTMA (certification):** The $\mathrm{SNR}(r)$ profile affects object manifold geometry.
  Hypothesis: the Watson-based SNR profile maximizes **manifold capacity** ($\alpha_c$) per
  unit compute; gives lower radius/dimension than flat blur and i.i.d. noise
  (see `experiment-plan-and-ablations.md §6`, Dapello–Chung [22]).

---

## 7. Mapping to ablations and measurement

This guide maps to the **C2 axis** and **A-group** in `experiment-plan-and-ablations.md`
as follows:

- **A2–A4:** three regimes compared at **equal average SNR budget** (isolates structure effect).
- **A5 → SNR profile sweep (new):** trace-based regime sweeps $\beta\in\{1,2,3\}$ and
  $\mathrm{SNR}_0$; compares "Watson-based $\mathrm{SNR}(r)$" vs "flat SNR" vs
  "arbitrary blur schedule".
- **Measured SNR metric:** the realized $\mathrm{SNR}(r)$ curve is reported for each
  condition (eccentricity-binned), so accuracy/robustness/capacity are plotted **as a
  function of SNR**. This is one of the thesis's main quantitative figures.

---

## 8. Implementation notes (with override)

Per `technical-guide.md §3`, this mechanism is added as a new `nn.Module` (`Periphery`)
without modifying existing libraries, plugged into the `FoveatedModel` wrapper:

- In `src/foveation.py`: `WatsonPooling(r)`, `SNRProfile(beta, snr0)`, `TraceBasedPeriphery(stat='adain'|'gram'|'ps', synth='amortized')`.
- Regime selection with a single parameter: `periphery_mode ∈ {'blur','iid','trace'}` → A2/A3/A4.
- Kept differentiable (for curriculum and optional end-to-end training); amortized synthesis
  network pre-trainable separately.
- Because it is stochastic: fix seed during evaluation; **EOT-N required for adversarial
  evaluation** (metameric sample changes on every forward pass).

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Synthesis cost eats foveation gain | Amortized small generative network (NeuroFovea/SideEye); optimization only for reference/validation |
| α–SNR relationship may deviate from model | Empirically calibrate $\rho$ (fit measured SNR curve to target) |
| Too low peripheral SNR → accuracy drops | Find SNR–accuracy sweet spot via $\beta$, $\mathrm{SNR}_0$ sweep |
| Adversarial without EOT → spurious robustness | EOT-N (N≥10) in all adversarial measurements |
| Watson formula depends on foveal ppd/viewing angle | Fix the image→degree conversion (visual_degrees, ppd) consistently with VOneNet |
