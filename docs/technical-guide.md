# Technical Guide — Experimental Infrastructure, Colab Setup, and Override Strategy

> **Thesis:** Biologically Inspired Vision Models — Attention-driven foveation,
> V1 front-end (VOneNet), recurrent ventral stream, and IT-feedback loop.
> **Purpose of this document:** Define the experimental environment (Colab), code
> organization, and how to **override** existing libraries to implement changes.

---

## 0. Core Principles (binding rules)

The entire experimental infrastructure of this thesis is built around three principles:

1. **Working environment is Google Colab.** All training/evaluation experiments are
   designed to run on Colab (preferably Colab Pro, A100/L4 GPU). Checkpoints for long
   ImageNet training runs are written to Google Drive; **resume-from-checkpoint** is
   mandatory in every notebook to handle session drops.

2. **All code is in notebook (`.ipynb`) format.** Each experiment is a self-contained,
   end-to-end runnable Jupyter notebook. `.py` modules are used only for shared utility
   functions and are imported from within notebooks; but **experiment logic and overrides
   must be explicitly visible inside notebook cells** (for reproducibility transparency).

3. **Existing libraries are not rewritten — they are overridden.** VOneNet, CORnet, FOVEA,
   etc. repos are cloned and added to `sys.path`. Required behavioral changes (e.g.,
   disabling V1 noise, prepending foveal transform, IT-feedback loop) are made by
   **overriding/monkey-patching the original classes**. Upstream code is never forked and
   hand-edited; this preserves source fidelity and lets ablations be toggled with a single
   flag.

---

## 1. Colab Working Environment

### 1.1 Standard opening cell (beginning of every notebook)

```python
# --- Environment detection: Colab or local? ---
import sys, os, subprocess
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = '/content/drive/MyDrive/tez_foveated_vision'
else:
    PROJECT_ROOT = os.path.expanduser('~/Desktop/Bogazici/Tez/foveated_vision')

os.makedirs(PROJECT_ROOT, exist_ok=True)
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)

# GPU check
import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

### 1.2 Dependencies

Every notebook contains a version-pinned installation cell. **Pin versions**
(prevents breakage when the Colab image changes):

```python
!pip -q install torch==2.* torchvision timm==1.* foolbox==3.* robustbench \
    scipy scikit-learn matplotlib tqdm
# Brain-Score only in the evaluation notebook:
# !pip -q install git+https://github.com/brain-score/vision.git
```

### 1.3 Adding repos to path (no forking)

```python
REPOS = {
    'vonenet': 'https://github.com/dicarlolab/vonenet.git',
    'CORnet':  'https://github.com/dicarlolab/CORnet.git',
    'fovea':   'https://github.com/tchittesh/fovea.git',
}
REPO_DIR = os.path.join(PROJECT_ROOT, 'external')
os.makedirs(REPO_DIR, exist_ok=True)
for name, url in REPOS.items():
    dst = os.path.join(REPO_DIR, name)
    if not os.path.exists(dst):
        subprocess.run(['git', 'clone', '--depth', '1', url, dst], check=True)
    if dst not in sys.path:
        sys.path.insert(0, dst)
```

> Repos are cloned under `external/`, not `data/`, and are **never edited**.
> This guarantees they stay identical to the official repos in `paper-repositories.md`.

---

## 2. Repository / Notebook Organization

The `tez_foveated_vision/` root on Google Drive:

```
tez_foveated_vision/
├── external/                 # cloned official repos (READ-ONLY, overridden)
│   ├── vonenet/  CORnet/  fovea/ ...
├── src/                      # shared utility .py modules (imported from notebooks)
│   ├── overrides.py          # all monkey-patch/override classes collected here
│   ├── foveation.py          # R-Blur, trace-based periphery transforms
│   ├── it_feedback.py        # confidence/uncertainty-driven fixation + halting
│   ├── mftma.py              # manifold capacity certification wrapper
│   └── eval_harness.py       # ImageNet-C, PGD/APGD, Brain-Score bridges
├── notebooks/                # ALL experiments as .ipynb (numbered, sequential)
│   ├── 00_setup_and_data.ipynb
│   ├── 01_baseline_reproduce.ipynb
│   ├── 02_foveation_rblur_and_periphery.ipynb
│   ├── 03_v1_block.ipynb
│   ├── 04_it_feedback_multiglance.ipynb
│   ├── 05_mftma_certification.ipynb
│   └── 06_full_model_and_ablations.ipynb
├── checkpoints/              # model weights (for resume)
└── results/                  # metric CSV/JSON + figures
```

**Notebook naming convention:** `NN_short_name.ipynb`. Each notebook is responsible
only for its own phase and writes its output under `results/`; the next notebook reads it.

---

## 3. Override Strategy (modifying libraries without rewriting them)

Most experiments require pointwise changes to the behavior of existing classes.
**Three patterns** are used for this. All of them are collected in `src/overrides.py`
and activated from a notebook with a single line.

### 3.1 Pattern A — Behavioral change via subclassing

The cleanest approach. Override only the needed method without touching the original class.
Example: making VOneBlock's Poisson noise toggleable for ablation.

```python
# src/overrides.py
from vonenet.modules import VOneBlock

class ConfigurableVOneBlock(VOneBlock):
    """VOneBlock + toggle noise on/off at runtime (ablation: V1 Poisson noise on/off)."""
    def __init__(self, *args, noise_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_enabled = noise_enabled

    def noise_f(self, x):
        if not self._noise_enabled:
            return self.noise(x)          # ReLU only, no stochastic term
        return super().noise_f(x)         # original neuronal/gaussian noise
```

### 3.2 Pattern B — Monkey-patch (change at runtime without touching source)

When subclassing is impractical (e.g., deep inside a back-end), the method is replaced
at runtime. Must be reversible with an ablation flag.

```python
import vonenet.modules as vm
_orig_noise_f = vm.VOneBlock.noise_f      # save original (for rollback)

def _no_noise(self, x):
    return self.noise(x)

def set_v1_noise(enabled: bool):
    vm.VOneBlock.noise_f = _orig_noise_f if enabled else _no_noise
```

> **Rule:** Every monkey-patch must have a restore path so ablation conditions can
> be cleanly switched within a notebook.

### 3.3 Pattern C — Adding a new component via wrapping

The thesis's **original contributions** — foveation, trace-based periphery, and
IT-feedback — do not exist upstream; they are added as new `nn.Module` wrappers around
the existing model. VOneNet already has a `nn.Sequential(vone_block, bottleneck,
model_back_end)` structure (`vonenet/vonenet.py`); we insert the foveal transform
**before** this sequence and the IT-feedback loop **around** it.

```python
import torch.nn as nn

class FoveatedModel(nn.Module):
    """Foveal transform + (VOneNet body) + IT-feedback loop — all as a wrapper."""
    def __init__(self, core, foveate, periphery, it_head=None):
        super().__init__()
        self.foveate = foveate       # attention -> fixation -> R-Blur foveal warp
        self.periphery = periphery   # trace-based metameric noise (original contribution)
        self.core = core             # overridden VOneNet/ResNet body
        self.it_head = it_head       # confidence/uncertainty + halting (optional)

    def forward(self, x, fixation=None, max_glances=1):
        logits = None
        for t in range(max_glances):
            fov = self.foveate(x, fixation)
            comp = self.periphery(x, fov, fixation)   # sharp fovea + metameric periphery
            logits, conf, unc = self._readout(comp)
            if self.it_head is None or self.it_head.should_halt(conf):
                break
            fixation = self.it_head.next_fixation(x, conf, unc)  # IT feedback
        return logits

    def _readout(self, comp):
        out = self.core(comp)
        if self.it_head is not None:
            return self.it_head(out)
        return out, None, None
```

> Note: Due to `nn.DataParallel` wrapping (inside `get_model`), access the actual
> model via `model.module` before applying overrides; attach wrappers at the `module` level.

### 3.4 Override map (which ablation → which override)

| Thesis component / ablation | Target class | Override pattern |
|-----------------------------|-------------|------------------|
| V1 Poisson noise on/off | `vonenet.modules.VOneBlock.noise_f` | A or B |
| Foveation on/off (R-Blur) | `Foveate` module prepended to input | C (wrapper) |
| Periphery regime (blur / i.i.d. / trace-based) | `Periphery` module, strategy parameter | C |
| IT feedback on/off, multi-glance | `FoveatedModel` loop + `it_head` | C |
| Back-end: ResNet-50 ↔ CORnet-S | `vonenet.vonenet.VOneNet(model_arch=...)` | existing parameter |
| Curriculum vs fixed blur | blur scheduler in training loop | notebook training cell |

---

## 4. Reproducibility Rules

Every notebook must include:

- **Fixed seed:** `torch.manual_seed`, `numpy.random.seed`, `random.seed` + cudnn
  deterministic. Note: use `vone_block.fix_noise(seed=...)` to fix VOneNet's stochastic
  front-end during evaluation (see `modules.py: fix_noise/unfix_noise`).
- **Config block:** all hyperparameters in a single `CFG` dict; written to the result
  file as JSON.
- **Checkpoint + resume:** save to Drive after each epoch; load if already present.
- **Result logging:** metrics in `results/NN_*.json` + figures in `results/NN_*.png`.
- **Environment stamp:** `torch.__version__`, GPU name, repo commit hashes added to the
  result JSON.

---

## 5. Evaluation Bridges (eval harness)

Collected in `src/eval_harness.py`, called from notebooks:

- **Clean accuracy:** ImageNet-1K / CIFAR-10 val.
- **Corruption robustness:** ImageNet-C (15 corruptions × 5 severities) → mCE;
  ImageNet-R/Sketch.
- **Adversarial:** PGD and APGD (foolbox / robustbench); for stochastic front-ends use
  **EOT (Expectation over Transformation)** for gradients — otherwise noise produces
  spurious robustness.
- **Shape bias / cue-conflict:** Geirhos stylized-ImageNet dataset.
- **Brain-Score:** V1/V2/V4/IT predictivity; local public benchmark + optional submission.
- **MFTMA:** `src/mftma.py` (see `neural_manifolds_replicaMFT`) — capacity/radius/dimension.

> **Critical warning (stochastic models):** VOneNet and trace-based periphery are
> stochastic. If EOT is not used in adversarial evaluation, robustness numbers are
> artificially inflated. All adversarial ablations are reported with EOT-N (N≥10).

---

## 6. Summary Workflow

1. `00_setup_and_data.ipynb` → Drive mount, clone repos, prepare datasets.
2. `01_baseline_reproduce.ipynb` → VOneResNet-50 + ResNet-50 baseline, validate on 1
   robustness slice.
3. `02..04` → add foveation, V1 front-end, IT-feedback components **via override**.
4. `05_mftma_certification.ipynb` → geometric certification.
5. `06_full_model_and_ablations.ipynb` → full closed-loop model + all ablations from
   `experiment-plan-and-ablations.md`.

For the detailed ablation matrix and methodologies: **`experiment-plan-and-ablations.md`**.
For repository links: **`paper-repositories.md`**.
