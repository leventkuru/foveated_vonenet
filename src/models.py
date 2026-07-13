"""Baseline model builders for B0-B3 (ResNet-50, AlexNet, VOneResNet-50, CORnet-S).

Each builder returns `(model, normalization)`, so callers always pair a model with the
normalization convention it was trained under (see datasets.py Sec. "normalization
conventions"): "imagenet" for B0/B1/B3, "vonenet" for B2.

In smoke_test mode, pass `pretrained=False` -- every builder then falls back to a
randomly-initialized model with no network access, so the notebook's *pipeline* can be
verified quickly and independent of network conditions. Any accuracy computed on a
randomly-initialized model is meaningless by construction and must never be reported as
a real baseline number.

Requires `external/vonenet` and `external/CORnet` to already be on `sys.path` (done by
00_setup_and_data.ipynb's repo-cloning cell).
"""

import torch
import torch.nn as nn
import torchvision

from . import datasets as _datasets


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip an `nn.DataParallel` wrapper if present. VOneNet and CORnet-S builders
    both return DataParallel-wrapped models (matching their upstream `get_model`
    conventions), so this is needed to reach the underlying architecture (e.g. to find
    `vone_block` for `fix_noise`)."""
    return model.module if isinstance(model, nn.DataParallel) else model


def build_resnet50(pretrained: bool = True):
    """Baseline B0: plain torchvision ResNet-50, ImageNet normalization."""
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    return model, "imagenet"


def build_alexnet(pretrained: bool = True):
    """Baseline B1: plain torchvision AlexNet, ImageNet normalization."""
    weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.alexnet(weights=weights)
    return model, "imagenet"


def build_voneresnet50(pretrained: bool = True, map_location: str = "cpu"):
    """Baseline B2: VOneNet with a ResNet-50 back-end (Dapello et al., 2020).

    VOneNet normalization (mean=std=0.5). Structure is
    `DataParallel(Sequential(vone_block, bottleneck, model))` -- see
    `external/vonenet/vonenet/vonenet.py`. `pretrained=True` downloads weights from
    the VOneNet S3 bucket (see `external/vonenet/vonenet/__init__.py::get_model`);
    `pretrained=False` builds the same architecture with fresh (untrained) weights and
    performs no network access.

    NOTE: gradient-based attacks (PGD/EOT) against the returned model require
    `overrides.apply_vone_block_input_gradient_fix()` to be active first -- the
    unpatched `VOneBlock.gabors_f` raises a RuntimeError when a gradient is taken with
    respect to the model's *input* (see `src/overrides.py` for the full root-cause
    analysis). This function does not apply the patch itself, so that overrides stay
    visible in the calling notebook rather than hidden inside a builder function.

    NOTE: `vonenet.get_model()` always wraps the model in `nn.DataParallel`, whose
    `device_ids` are inferred from `torch.cuda.is_available()` at construction time --
    independent of `map_location`. On a CUDA-visible machine with the default
    `map_location="cpu"`, this leaves `device_ids=[0]` (expects cuda:0) while the
    actual parameters sit on CPU, so `DataParallel.forward()` raises "module must have
    its parameters and buffers on device cuda:0 ... but found ... cpu" on the very
    first call, regardless of what device the input tensor is on. `_detach_data_parallel`
    neutralizes this below: it moves the model to the caller's requested
    `map_location` (never silently overriding an explicit "cpu" choice) and clears
    `device_ids` so `DataParallel.forward()` skips the (buggy) device check entirely
    and just calls the wrapped module directly -- correct for the single-device
    inference/training this project does, and harmless if the caller later moves the
    model again with `.to(...)`.
    """
    import vonenet
    model = vonenet.get_model(model_arch="resnet50", pretrained=pretrained, map_location=map_location)
    return _detach_data_parallel(model, map_location), "vonenet"


def build_cornet_s(pretrained: bool = True, map_location: str = "cpu"):
    """Baseline B3: standalone CORnet-S (Kubilius et al., 2019) -- the recurrent
    ventral-stream reference, with no VOneBlock front-end. ImageNet normalization
    (confirmed against `external/CORnet/run.py`'s training/eval transforms). Structure
    is `DataParallel(Sequential(V1, V2, V4, IT, decoder))` -- see
    `external/CORnet/cornet/cornet_s.py`. `pretrained=True` downloads weights from the
    CORnet S3 bucket; `pretrained=False` performs no network access.

    NOTE: like `build_voneresnet50`, `cornet.cornet_s()` wraps the model in
    `nn.DataParallel` whose `device_ids` follow `torch.cuda.is_available()`, but never
    itself moves the underlying parameters off CPU (`map_location` there only affects
    where the *downloaded checkpoint* tensors land before `load_state_dict` copies
    their values in-place). So on a CUDA-visible machine the same device_ids-vs-actual-
    device mismatch as VOneNet occurs, independent of `map_location`. See
    `_detach_data_parallel`.
    """
    import cornet
    model = cornet.cornet_s(pretrained=pretrained, map_location=map_location)
    return _detach_data_parallel(model, map_location), "imagenet"


def _detach_data_parallel(model: nn.Module, map_location) -> nn.Module:
    """Move `model` to `map_location` and, if it's `nn.DataParallel`-wrapped, clear
    `device_ids` so `DataParallel.forward()` skips its device-consistency check and
    calls the wrapped module directly (see the NOTE in `build_voneresnet50` /
    `build_cornet_s` for why that check is otherwise unreliable here). Single-device
    inference/training never needed the multi-GPU replication DataParallel provides,
    so this is a pure bugfix, not a behavior change for this project's use.
    """
    model = model.to(map_location)
    if isinstance(model, nn.DataParallel):
        model.device_ids = []
    return model


BASELINE_BUILDERS = {
    "resnet50": build_resnet50,
    "alexnet": build_alexnet,
    "voneresnet50": build_voneresnet50,
    "cornet_s": build_cornet_s,
}


def get_baseline(name: str, pretrained: bool = True):
    """Dispatch to the builder for baseline `name` in {resnet50, alexnet,
    voneresnet50, cornet_s}. Returns (model, normalization)."""
    if name not in BASELINE_BUILDERS:
        raise ValueError(f"Unknown baseline {name!r}; choices are {list(BASELINE_BUILDERS)}")
    return BASELINE_BUILDERS[name](pretrained=pretrained)


def get_vone_block(model: nn.Module):
    """Return the `VOneBlock` submodule of a (possibly DataParallel-wrapped) VOneNet
    model, or None if `model` has no such submodule (i.e. it is not a VOneNet model).
    Used to fix/unfix the stochastic front-end's noise for deterministic evaluation
    (`fix_noise`) or fresh-per-sample EOT gradients (`unfix_noise`)."""
    core = unwrap_model(model)
    return getattr(core, "vone_block", None)


class NormalizedModel(nn.Module):
    """Wraps `model` so it accepts raw [0, 1]-range pixel tensors and applies `model`'s
    own (mean, std) normalization internally before the forward pass.

    Adversarial attacks (PGD/APGD) are run against this wrapper on raw pixel tensors
    (via a `normalization="raw"` dataset transform, see datasets.py), so that an
    epsilon budget in [0, 1] pixel units is directly comparable across baselines that
    use different input-normalization conventions (B0/B1/B3 vs. B2).
    """

    def __init__(self, model: nn.Module, normalization: str):
        super().__init__()
        self.model = model
        mean, std = _datasets.normalization_stats(normalization)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def wrap_for_raw_pixel_input(model: nn.Module, normalization: str) -> NormalizedModel:
    """Convenience wrapper around `NormalizedModel` -- see its docstring."""
    return NormalizedModel(model, normalization)
