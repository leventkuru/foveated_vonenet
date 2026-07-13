"""
Foveation transforms for the foveated-vision thesis.

Public API
----------
WatsonPoolingScale   -- mRGC receptive-field spacing s(r) [Watson 2014, J.Vis 14(7):15]
SNRProfile           -- SNR(r) = SNR0 * (s0/s(r))^beta with alpha(r) knob
eccentricity_map     -- pixel-distance eccentricity tensor (pixels + degrees)
gaussian_blur_2d     -- isotropic Gaussian convolution (depthwise, reflect pad)
soft_foveal_mask     -- sigmoid soft-mask: ~1 inside fovea, ~0 in periphery
apply_rblur          -- R-Blur: eccentricity-dependent Gaussian blur (pyramid blend)
FlatBlurPeriphery    -- Ablation A2: low-pass Gaussian in periphery
IIDNoisePeriphery    -- Ablation A3: signal-independent Gaussian noise
TraceBasedPeriphery  -- Ablation A4 (original): signal-conditioned AdaIN metameric
FoveatedTransform    -- Composite: sharp fovea mask + chosen periphery module

References
----------
[9]  Narang & Bhatt 2020 (R-Blur / RetinalBlur)
[23] Watson 2014, J. Vision 14(7):15  (mRGC spacing formula)
TRACE_BASED_NOISE_REHBERI.md  (SNR formalisation, original contribution)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Watson (2014) mRGC pooling scale
# -----------------------------------------------------------------------

class WatsonPoolingScale:
    """
    Midget RGC receptive-field spacing as a function of eccentricity.

    Watson (2014) fits Curcio & Allen (1990) ganglion-cell density data and
    derives a near-linear relation between eccentricity r (degrees) and the
    mean mRGC inter-cell spacing s(r) (degrees):

        60 * s(r) ≈ 0.53 + 0.434 * r
    =>  s(r) = (0.53 + 0.434 * r) / 60   [degrees]

    At the fovea (r=0): s(0) = 0.53/60 ≈ 0.0088° ≈ 0.53 arcmin ≈ cone spacing.
    Pooling window radius: w(r) = kappa * s(r), kappa ≥ 1.
    """

    A = 0.434   # slope  [arcmin / degree]  (Watson 2014, Table 1 fit)
    B = 0.53    # intercept [arcmin] at r=0

    def spacing(self, r_deg):
        """s(r) in degrees. r_deg may be float, ndarray, or torch.Tensor."""
        return (self.B + self.A * r_deg) / 60.0

    def spacing_fovea(self):
        """s(0) in degrees — reference spacing at the fixation point."""
        return self.B / 60.0

    def window_radius(self, r_deg, kappa=1.0):
        """Pooling window radius w(r) = kappa * s(r) in degrees."""
        return kappa * self.spacing(r_deg)


# -----------------------------------------------------------------------
# SNR profile
# -----------------------------------------------------------------------

class SNRProfile:
    """
    Eccentricity-dependent SNR(r) derived from the Watson pooling scale.

    Formula (TRACE_BASED_NOISE_REHBERI.md Sec. 3.2):

        SNR(r) = SNR0 * (s(0) / s(r))^beta

    where s(r) is the Watson mRGC spacing, SNR0 is the foveal (r=0) SNR, and
    beta controls the roll-off steepness (beta=2 corresponds to area-based pooling).

    Content-texture interpolation parameter alpha(r) (Sec. 3.3):

        alpha(r) = sqrt(SNR(r)/rho) / (1 + sqrt(SNR(r)/rho))

    where rho is the signal-to-texture power ratio (default 1.0).
    """

    def __init__(self, snr0_db: float = 30.0, beta: float = 2.0,
                 ppd: float = 28.0, rho: float = 1.0):
        self.snr0 = 10.0 ** (snr0_db / 10.0)   # linear SNR at fovea
        self.beta = beta
        self.ppd = ppd
        self.rho = rho
        self._watson = WatsonPoolingScale()

    def snr(self, r_deg):
        """Linear SNR(r). r_deg may be float or torch.Tensor."""
        s0 = self._watson.spacing_fovea()
        sr = self._watson.spacing(r_deg)
        return self.snr0 * (s0 / sr) ** self.beta

    def snr_db(self, r_deg):
        """SNR(r) in decibels."""
        return 10.0 * torch.log10(self.snr(r_deg))

    def alpha(self, r_deg):
        """
        Content-texture interpolation coefficient alpha(r) in [0, 1].
        alpha -> 1 at fovea (full signal); alpha -> 0 far in periphery (pure texture).
        """
        snr_r = self.snr(r_deg)
        sqrt_term = torch.sqrt(snr_r / self.rho)
        return sqrt_term / (1.0 + sqrt_term)


# -----------------------------------------------------------------------
# Core geometry helpers
# -----------------------------------------------------------------------

def eccentricity_map(H: int, W: int, fixation_yx=(0.5, 0.5), ppd: float = 28.0):
    """
    Eccentricity tensor for an H x W image.

    Args:
        H, W:         image height and width in pixels.
        fixation_yx:  relative fixation (fy, fx) in [0, 1] x [0, 1].
        ppd:          pixels per degree of visual angle.

    Returns:
        r_pix: [H, W] eccentricity in pixels.
        r_deg: [H, W] eccentricity in degrees.
    """
    cy = fixation_yx[0] * H
    cx = fixation_yx[1] * W
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r_pix = torch.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
    r_deg = r_pix / ppd
    return r_pix, r_deg


def gaussian_blur_2d(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Isotropic Gaussian blur on a [C, H, W] or [B, C, H, W] tensor.
    Uses depthwise convolution with reflect padding.
    Returns the input unchanged if sigma < 1e-3.
    """
    if sigma < 1e-3:
        return image
    squeeze = image.dim() == 3
    if squeeze:
        image = image.unsqueeze(0)
    B, C, H, W = image.shape
    radius = max(1, min(int(math.ceil(3.0 * sigma)), min(H, W) - 1))
    ks = 2 * radius + 1
    xs = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    k1d = torch.exp(-0.5 * (xs / sigma) ** 2)
    k1d = k1d / k1d.sum()
    k2d = k1d[:, None] * k1d[None, :]  # [ks, ks]
    k2d = k2d.view(1, 1, ks, ks).expand(C, 1, ks, ks).contiguous()
    padded = F.pad(image, [radius] * 4, mode="replicate")
    blurred = F.conv2d(padded, k2d, groups=C)
    if squeeze:
        blurred = blurred.squeeze(0)
    return blurred


def soft_foveal_mask(H: int, W: int, fixation_yx=(0.5, 0.5),
                      fovea_deg: float = 1.5, transition_deg: float = 0.5,
                      ppd: float = 28.0) -> torch.Tensor:
    """
    Soft foveal mask in [0, 1]: ~1 inside the foveal region, ~0 in periphery.
    Uses a sigmoid transition:
        m(r) = sigmoid( -(r - fovea_deg) / transition_deg )
    """
    _, r_deg = eccentricity_map(H, W, fixation_yx, ppd)
    return torch.sigmoid(-(r_deg - fovea_deg) / transition_deg)


# -----------------------------------------------------------------------
# R-Blur: eccentricity-dependent Gaussian blur
# -----------------------------------------------------------------------

def apply_rblur(image: torch.Tensor,
                fixation_yx=(0.5, 0.5),
                sigma0: float = 0.5,
                slope: float = 1.5,
                ppd: float = 28.0,
                n_levels: int = 5) -> torch.Tensor:
    """
    R-Blur foveal transform: eccentricity-dependent Gaussian blur.

    Blur sigma at each pixel: sigma(r) = sigma0 + slope * r_deg  [pixels].
    Implemented via Gaussian pyramid blending with triangular window weights:
      1. Build N blurred versions at uniformly-spaced sigma levels.
      2. At each pixel, blend the two adjacent levels proportionally.

    Args:
        image:       [C, H, W] float tensor in [0, 1].
        fixation_yx: relative fixation (fy, fx) in [0, 1]^2.
        sigma0:      blur sigma at the fixation point (pixels).
        slope:       rate of sigma increase per degree of eccentricity (px/deg).
        ppd:         pixels per degree of visual angle.
        n_levels:    number of Gaussian pyramid levels.

    Returns: [C, H, W] float tensor in [0, 1].
    """
    C, H, W = image.shape
    device = image.device
    _, r_deg = eccentricity_map(H, W, fixation_yx, ppd)
    r_deg = r_deg.to(device)
    sigma_map = (sigma0 + slope * r_deg).clamp(min=0.0)  # [H, W]
    sigma_max = float(sigma_map.max().item())
    sigma_levels = torch.linspace(0.0, sigma_max, n_levels)
    # Build pyramid
    blurred_stack = torch.stack(
        [gaussian_blur_2d(image, float(s.item())) for s in sigma_levels], dim=0
    )  # [N, C, H, W]
    # Triangular blending weights at each pixel
    level_span = sigma_levels[1] - sigma_levels[0] + 1e-8  # uniform spacing
    dist = (sigma_map[None, :, :] - sigma_levels[:, None, None]).abs() / level_span
    weights = F.relu(1.0 - dist)           # [N, H, W]
    weights = weights / (weights.sum(0, keepdim=True) + 1e-8)
    output = (weights[:, None, :, :] * blurred_stack).sum(0)  # [C, H, W]
    return output.clamp(0.0, 1.0)


# -----------------------------------------------------------------------
# Periphery regime modules (A2, A3, A4)
# -----------------------------------------------------------------------

class FlatBlurPeriphery(nn.Module):
    """
    Ablation A2: Low-pass Gaussian blur in the periphery (no added noise).
    Implements R-Blur with a potentially larger slope to match SNR budget.
    """

    def __init__(self, sigma0: float = 0.5, slope: float = 2.0,
                 ppd: float = 28.0, fixation_yx=(0.5, 0.5)):
        super().__init__()
        self.sigma0 = sigma0
        self.slope = slope
        self.ppd = ppd
        self.fixation_yx = fixation_yx

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return apply_rblur(image, self.fixation_yx, self.sigma0,
                           self.slope, self.ppd)


class IIDNoisePeriphery(nn.Module):
    """
    Ablation A3: Signal-independent i.i.d. Gaussian noise in the periphery.

    Noise amplitude is derived from the SNR profile so that at each eccentricity r
    the added noise power matches the target SNR(r):

        sigma_noise(r) = sqrt( Var[I] / SNR(r) )

    This makes A3 directly comparable to A4 (trace-based) at equal SNR budget.
    """

    def __init__(self, snr_profile: SNRProfile,
                 fixation_yx=(0.5, 0.5), ppd: float = 28.0):
        super().__init__()
        self.snr_profile = snr_profile
        self.fixation_yx = fixation_yx
        self.ppd = ppd

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        C, H, W = image.shape
        device = image.device
        _, r_deg = eccentricity_map(H, W, self.fixation_yx, self.ppd)
        r_deg = r_deg.to(device)
        snr_map = self.snr_profile.snr(r_deg).clamp(min=1e-3)  # [H, W]
        signal_var = image.var().clamp(min=1e-6)
        sigma_map = torch.sqrt(signal_var / snr_map)   # [H, W]
        noise = torch.randn_like(image)
        return (image + sigma_map[None] * noise).clamp(0.0, 1.0)


class TraceBasedPeriphery(nn.Module):
    """
    Ablation A4 — Original contribution: signal-conditioned AdaIN metameric noise.

    Peripheral output at pixel p with eccentricity r:

        I_hat(p) = alpha(r) * I(p) + (1 - alpha(r)) * T(S_{Omega(r)}(I))(p)

    where:
      * S_{Omega} = local AdaIN statistics (mean, std) over a patch of size
        proportional to the Watson pooling window (kappa * s(r) * ppd pixels).
      * T(S) = AdaIN texture sample: Gaussian noise rescaled to match local
        mean and std — this preserves the *trace* (local statistics) while
        randomising the fine spatial structure (phase).
      * alpha(r) from SNRProfile.alpha() — encodes Watson-grounded SNR decay.

    This is differentiable and can be used end-to-end in training.
    For adversarial evaluation, fix seed and use EOT-N.
    """

    def __init__(self, snr_profile: SNRProfile,
                 patch_size: int = 8,
                 fixation_yx=(0.5, 0.5),
                 ppd: float = 28.0):
        super().__init__()
        self.snr_profile = snr_profile
        self.patch_size = patch_size
        self.fixation_yx = fixation_yx
        self.ppd = ppd

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        C, H, W = image.shape
        device = image.device
        _, r_deg = eccentricity_map(H, W, self.fixation_yx, self.ppd)
        r_deg = r_deg.to(device)
        alpha_map = self.snr_profile.alpha(r_deg)         # [H, W]
        texture = self._adain_texture(image)               # [C, H, W]
        return (alpha_map[None] * image
                + (1.0 - alpha_map[None]) * texture).clamp(0.0, 1.0)

    def _adain_texture(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate an AdaIN texture sample matching local patch statistics.
        Steps:
          1. Pad image to next multiple of patch_size.
          2. Extract non-overlapping patches [C, nH, nW, p, p].
          3. Compute patch-level (mean, std).
          4. Normalise a random Gaussian draw, then rescale/shift to match.
          5. Reconstruct and crop to original size.
        """
        C, H, W = image.shape
        device = image.device
        p = self.patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        img_p = F.pad(image, [0, pad_w, 0, pad_h], mode="reflect")  # [C, Hp, Wp]
        noise_p = torch.randn(C, H + pad_h, W + pad_w,
                              device=device, dtype=image.dtype)
        # Extract patches: [C, nH, nW, p, p]
        img_unf = img_p.unfold(1, p, p).unfold(2, p, p)
        noise_unf = noise_p.unfold(1, p, p).unfold(2, p, p)
        mu = img_unf.mean(dim=(-2, -1), keepdim=True)
        sigma = img_unf.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        noise_mu = noise_unf.mean(dim=(-2, -1), keepdim=True)
        noise_std = noise_unf.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        adain_unf = sigma * (noise_unf - noise_mu) / noise_std + mu
        # Reconstruct [C, Hp, Wp] from [C, nH, nW, p, p]
        nH = (H + pad_h) // p
        nW = (W + pad_w) // p
        texture = adain_unf.permute(0, 1, 3, 2, 4).reshape(C, nH * p, nW * p)
        return texture[:, :H, :W].clamp(0.0, 1.0)


# -----------------------------------------------------------------------
# Composite foveated transform
# -----------------------------------------------------------------------

class FoveatedTransform(nn.Module):
    """
    Composite foveated image transform.

    Output = m(r) * I + (1 - m(r)) * Periphery(I)

    where m(r) is the soft foveal mask (≈1 at fixation, ≈0 far periphery).
    The foveal region always sees the sharp, unmodified input image.

    Args:
        periphery:      nn.Module from {FlatBlurPeriphery, IIDNoisePeriphery,
                        TraceBasedPeriphery}, or None (no-op, A1 baseline).
        fovea_deg:      fovea radius in degrees.
        transition_deg: soft-mask transition width in degrees.
        ppd:            pixels per degree.
        fixation_yx:    relative fixation (fy, fx) in [0,1]^2.
    """

    def __init__(self, periphery: nn.Module = None,
                 fovea_deg: float = 1.5,
                 transition_deg: float = 0.5,
                 ppd: float = 28.0,
                 fixation_yx=(0.5, 0.5)):
        super().__init__()
        self.periphery = periphery
        self.fovea_deg = fovea_deg
        self.transition_deg = transition_deg
        self.ppd = ppd
        self.fixation_yx = fixation_yx

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        C, H, W = image.shape
        device = image.device
        mask = soft_foveal_mask(H, W, self.fixation_yx,
                                self.fovea_deg, self.transition_deg,
                                self.ppd).to(device)   # [H, W]
        if self.periphery is None:
            return image
        periph = self.periphery(image)  # [C, H, W]
        return (mask[None] * image + (1.0 - mask[None]) * periph).clamp(0.0, 1.0)


def build_foveated_transform(mode: str, snr_profile: SNRProfile,
                              patch_size: int = 8,
                              fovea_deg: float = 1.5,
                              transition_deg: float = 0.5,
                              ppd: float = 28.0,
                              fixation_yx=(0.5, 0.5),
                              blur_sigma0: float = 0.5,
                              blur_slope: float = 2.0) -> FoveatedTransform:
    """
    Factory: build a FoveatedTransform for ablation mode in
    {'none', 'rblur', 'blur', 'iid', 'trace'}.
    """
    if mode == "none":
        periphery = None
    elif mode == "rblur":
        periphery = FlatBlurPeriphery(blur_sigma0, blur_slope, ppd, fixation_yx)
    elif mode == "blur":
        periphery = FlatBlurPeriphery(blur_sigma0, blur_slope, ppd, fixation_yx)
    elif mode == "iid":
        periphery = IIDNoisePeriphery(snr_profile, fixation_yx, ppd)
    elif mode == "trace":
        periphery = TraceBasedPeriphery(snr_profile, patch_size, fixation_yx, ppd)
    else:
        raise ValueError(f"Unknown periphery mode: {mode!r}")
    return FoveatedTransform(periphery, fovea_deg, transition_deg, ppd, fixation_yx)