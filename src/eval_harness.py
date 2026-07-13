"""Evaluation utilities shared across notebooks.

This module grows incrementally. 00_setup_and_data.ipynb only needed the ImageNet-C
corruption-error math. 01_baseline_reproduce.ipynb adds clean-accuracy evaluation and a
PGD attack with optional Expectation-over-Transformation (EOT) support, needed to
evaluate the stochastic VOneResNet-50 baseline (B2) fairly. Brain-Score bridging and
shape-bias evaluation are added alongside the notebooks that first need them
(03_v1_block.ipynb / 05_mftma_certification.ipynb).
"""

import torch
import torch.nn.functional as F

from .datasets import CORRUPTION_TYPES, SEVERITY_LEVELS


def corruption_error(model_errors: dict, baseline_errors: dict) -> dict:
    """Per-corruption Corruption Error CE_c (Hendrycks & Dietterich, 2019).

        CE_c = sum_s E_{s,c} / sum_s E_{s,c}^AlexNet

    Args:
        model_errors: {corruption_name: {severity: top1_error_rate}} for the
            evaluated model, top1_error_rate in [0, 1].
        baseline_errors: same shape, for the fixed AlexNet reference model used to
            normalize CE_c (so CE_c = 1 means "as robust as AlexNet").

    Returns:
        {corruption_name: CE_c} for every corruption present in both inputs.
    """
    ce = {}
    for corruption in model_errors:
        if corruption not in baseline_errors:
            continue
        model_sum = sum(model_errors[corruption][s] for s in SEVERITY_LEVELS)
        baseline_sum = sum(baseline_errors[corruption][s] for s in SEVERITY_LEVELS)
        ce[corruption] = model_sum / baseline_sum
    return ce


def mean_corruption_error(ce: dict) -> float:
    """mCE = (1 / 15) * sum_c CE_c, averaged over the 15 corruption types."""
    values = [ce[c] for c in CORRUPTION_TYPES if c in ce]
    if not values:
        raise ValueError("No matching corruption types found in `ce`.")
    return sum(values) / len(values)


@torch.no_grad()
def evaluate_clean_accuracy(model, dataset, device="cpu", batch_size=16, max_samples=None):
    """Top-1 accuracy of `model` on `dataset`.

    `dataset` should already yield tensors normalized for `model` (i.e. built with the
    correct `normalization` in `datasets.build_transform` / `build_cifar_transform`).
    `max_samples` caps the number of evaluated images (useful for a quick robustness
    slice or a smoke test); None evaluates the full dataset.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    correct, total = 0, 0
    for x, y in loader:
        if max_samples is not None and total >= max_samples:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return correct / total if total > 0 else float("nan")


def pgd_attack(model, x, y, eps, alpha, steps, eot_samples=1):
    """L-infinity PGD attack, with optional Expectation-over-Transformation (EOT).

        x_{t+1} = Proj_{||delta||_inf <= eps} ( x_t + alpha * sign(g_bar) )
        g_bar   = (1/N) * sum_i  grad_x  loss( model(x_t), y )      [N fresh forward
                                                                       passes per step]

    `model` must accept raw [0, 1]-range pixel tensors (see
    `models.wrap_for_raw_pixel_input`), so `eps`/`alpha` are in true pixel units and
    directly comparable across baselines with different input-normalization
    conventions.

    `eot_samples` (N above) must be > 1 for a *stochastic* model (VOneResNet-50, B2):
    with N=1, the single noisy forward/backward pass gives a high-variance gradient
    estimate that under-estimates the true attack strength -- a form of gradient
    masking that would make the model look more robust than it is (see
    TEKNIK_REHBER.md Sec. 5, "Kritik uyari"). Deterministic baselines (B0/B1/B3) can use
    eot_samples=1.
    """
    x = x.clone().detach()
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        grad_sum = torch.zeros_like(x_adv)
        for _ in range(eot_samples):
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
            grad, = torch.autograd.grad(loss, x_adv)
            grad_sum = grad_sum + grad
        grad_mean = grad_sum / eot_samples
        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * grad_mean.sign()
            delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
            x_adv = torch.clamp(x_orig + delta, min=0.0, max=1.0)
    return x_adv.detach()


def evaluate_pgd_robust_accuracy(model, dataset, device="cpu", eps=8 / 255, alpha=2 / 255,
                                  steps=10, eot_samples=1, batch_size=8, max_samples=None):
    """Top-1 accuracy of `model` under the PGD attack above.

    `model` must accept raw [0, 1] pixel input (see `pgd_attack`'s docstring), so build
    `dataset` with `normalization="raw"`.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    correct, total = 0, 0
    for x, y in loader:
        if max_samples is not None and total >= max_samples:
            break
        x, y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps, eot_samples=eot_samples)
        with torch.no_grad():
            logits = model(x_adv)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return correct / total if total > 0 else float("nan")
