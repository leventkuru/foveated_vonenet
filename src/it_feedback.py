"""
IT feedback and multi-glance loop components.

Public API
----------
FixationGrid           -- regular grid of fixation locations
MultiGlanceFoveatedModel -- D2: differentiable multi-glance (confidence halting)
FixationPolicy         -- D3: learned REINFORCE fixation policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.foveation import apply_rblur


class FixationGrid:
    """Regular fixation grid, centre fixation always first."""
    def __init__(self, n_rows=2, n_cols=2, margin=0.25):
        self.n_rows = n_rows; self.n_cols = n_cols; self.margin = margin
    def get_fixations(self):
        m = self.margin
        ys = [m + i / max(self.n_rows-1, 1)*(1-2*m) for i in range(self.n_rows)]
        xs = [m + j / max(self.n_cols-1, 1)*(1-2*m) for j in range(self.n_cols)]
        grid = [(y, x) for y in ys for x in xs]
        return sorted(grid, key=lambda p: abs(p[0]-0.5)+abs(p[1]-0.5))


class MultiGlanceFoveatedModel(nn.Module):
    """
    Differentiable multi-glance foveated classifier (D2).
    Backbone receives raw [0,1] pixel tensors.
    """
    def __init__(self, backbone_fn, fixations, halt_threshold=0.75,
                 sigma0=0.5, slope=1.5, ppd=4.0):
        super().__init__()
        self.backbone        = backbone_fn
        self.fixations       = fixations
        self.halt_threshold  = halt_threshold
        self.sigma0 = sigma0; self.slope = slope; self.ppd = ppd
    def foveate(self, x, fixation_yx):
        return torch.stack([apply_rblur(x[i], fixation_yx, self.sigma0,
                                        self.slope, self.ppd) for i in range(x.size(0))])
    def forward(self, x_raw, return_glance_count=False):
        B = x_raw.size(0)
        logit_sum = None; halted = torch.zeros(B, dtype=torch.bool, device=x_raw.device)
        glance_count = torch.ones(B, dtype=torch.long, device=x_raw.device)
        for t, fix in enumerate(self.fixations):
            fov_x = self.foveate(x_raw, fix)
            z_t   = self.backbone(fov_x)
            logit_sum = z_t if logit_sum is None else logit_sum + z_t
            if not self.training and t < len(self.fixations) - 1:
                conf = F.softmax(logit_sum / (t+1), dim=1).max(1).values
                newly = (conf >= self.halt_threshold) & (~halted)
                glance_count += (~halted & ~newly).long()
                halted |= newly
                if halted.all(): break
        final = logit_sum / len(self.fixations)
        return (final, glance_count) if return_glance_count else final


class FixationPolicy(nn.Module):
    """REINFORCE policy: state (logits) -> categorical over fixation locations."""
    def __init__(self, logit_dim, n_fixations):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(logit_dim, 32), nn.ReLU(), nn.Linear(32, n_fixations))
        self.baseline = 0.0
    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)