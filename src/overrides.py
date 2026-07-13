"""Monkey-patch / subclass overrides applied to code cloned into external/ -- the
source there is never edited directly (TEKNIK_REHBER.md Sec. 3). Every override here
is a single reversible flag: apply_*() patches, restore_*() reverts.

fix_vone_block_input_gradient_bug()
------------------------------------
`external/vonenet/vonenet/modules.py::VOneBlock.gabors_f` computes the complex-cell
response first,

    c = complex_nonlin(sqrt(s_q0[:, N_s:, :, :]**2 + s_q1[:, N_s:, :, :]**2) / sqrt(2))

then the simple-cell response,

    s = simple_nonlin(s_q0[:, :N_s, :, :])       # simple_nonlin = nn.ReLU(inplace=True)

`s_q0[:, N_s:, ...]` and `s_q0[:, :N_s, ...]` are two disjoint channel-slice *views* of
the same underlying `s_q0` tensor. Applying the in-place ReLU to the "simple" slice
bumps PyTorch's mutation version counter for `s_q0`'s *entire* storage (version
tracking is per-storage, not per-region), which invalidates the "complex" slice's
saved-for-backward values -- even though the two slices never physically overlap.

The upstream code was written and evaluated (Dapello et al., 2020) in a way that never
needed gradients *with respect to VOneBlock's input* (only w.r.t. parameters, for
training), so this was presumably never triggered there. It reproducibly breaks under
current PyTorch (verified: torch 2.6.0) the moment a gradient is taken w.r.t. the
input -- which is exactly what every adversarial attack in this thesis needs
(`eval_harness.pgd_attack`, used starting in 01_baseline_reproduce.ipynb for baseline
B2, and every ablation notebook that evaluates a VOneBlock-containing model under PGD).

Root-caused with `torch.autograd.set_detect_anomaly(True)`, which pins the failure to
the `PowBackward0` node created by `s_q0[:, N_s:, :, :] ** 2`.

Fix: `.clone()` the "simple" slice before the in-place ReLU, giving it its own storage
so the ReLU's mutation no longer touches `s_q0`'s version counter. Behaviorally
identical to the original on the forward pass (clone copies values, does not change
them); the only difference is which storage the in-place op is allowed to mutate.
"""

import numpy as np
import torch

import vonenet.modules as _vonenet_modules

_ORIGINAL_GABORS_F = _vonenet_modules.VOneBlock.gabors_f


def _patched_gabors_f(self, x):
    """Drop-in replacement for `VOneBlock.gabors_f` -- identical computation, with one
    added `.clone()` before the in-place ReLU (see module docstring for why)."""
    s_q0 = self.simple_conv_q0(x)
    s_q1 = self.simple_conv_q1(x)
    c = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                 s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
    s = self.simple(s_q0[:, 0:self.simple_channels, :, :].clone())
    return self.gabors(self.k_exc * torch.cat((s, c), 1))


def apply_vone_block_input_gradient_fix():
    """Patch `VOneBlock.gabors_f` at the class level (affects every VOneBlock instance,
    including ones already constructed, since it patches the class' bound method).
    Idempotent -- calling this more than once does not re-wrap the patch."""
    if _vonenet_modules.VOneBlock.gabors_f is not _patched_gabors_f:
        _vonenet_modules.VOneBlock.gabors_f = _patched_gabors_f


def restore_vone_block_input_gradient_fix():
    """Revert to the original, unpatched `VOneBlock.gabors_f`."""
    _vonenet_modules.VOneBlock.gabors_f = _ORIGINAL_GABORS_F


def is_vone_block_input_gradient_fix_applied() -> bool:
    return _vonenet_modules.VOneBlock.gabors_f is _patched_gabors_f


# ── V1 noise ablation (added by 03_v1_block.ipynb) ─────────────────────────

def set_v1_noise_mode(model, mode=None, noise_scale=1, noise_level=1):
    """Enable/disable VOneBlock neuronal noise for ablation B8 / B9.

    mode: None       -> deterministic Gabor bank, no noise  (B8)
          'neuronal' -> Poisson-like noise  (B9)
    Returns the VOneBlock instance. Idempotent.
    """
    from src.models import get_vone_block
    block = get_vone_block(model)
    if block is None:
        raise ValueError('Model has no VOneBlock.')
    block.set_noise_mode(mode, noise_scale, noise_level)
    return block


def is_v1_noise_active(model):
    """Return True if the VOneBlock noise is currently enabled."""
    from src.models import get_vone_block
    block = get_vone_block(model)
    return block is not None and block.noise_mode is not None
