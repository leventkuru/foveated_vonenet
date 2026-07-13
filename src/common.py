"""Cross-notebook utilities: seeding, config/result I/O, checkpoints, environment stamping.

Used by every notebook starting with 00_setup_and_data.ipynb. This module intentionally
holds no experiment logic or model overrides -- the Colab bootstrap cell (Drive mount,
path setup) is deliberately duplicated inline at the top of every notebook instead of
being wrapped here, so each notebook stays self-contained and readable top to bottom
(TEKNIK_REHBER.md Sec. 1.1 and Sec. 2).
"""

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs and force deterministic cuDNN kernels.

    Required by the reproducibility rules in TEKNIK_REHBER.md Sec. 4: every notebook
    must be able to reproduce its own numbers given the same CFG. Note this seeds the
    *global* RNG state; VOneNet's own stochastic front-end has a separate mechanism
    (`vone_block.fix_noise(seed=...)`) used starting in 01_baseline_reproduce.ipynb.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_config(cfg: dict, path: str) -> None:
    """Persist a notebook's CFG dict next to its results (TEKNIK_REHBER.md Sec. 4)."""
    save_json(cfg, path)


def save_figure(fig, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def save_checkpoint(state: dict, path: str) -> None:
    """Generic checkpoint writer.

    `state` typically holds model/optimizer state_dicts, the epoch index and the CFG
    used to produce them, so training can resume after a Colab session drop
    (TEKNIK_REHBER.md Sec. 1: 'resume-from-checkpoint zorunludur').
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None):
    """Returns the saved state dict, or None if no checkpoint exists yet (fresh run)."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=map_location)


def get_git_commit_hash(repo_dir: str):
    """Best-effort commit hash for provenance in the environment stamp.

    Returns None if `repo_dir` is not a git checkout or git is unavailable (e.g. after
    a plain zip upload to Colab) -- this must never raise.
    """
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def environment_stamp(project_root: str) -> dict:
    """Snapshot of the running environment.

    Saved into every notebook's results/*.json so numbers can be traced back to the
    exact torch/GPU/commit combination that produced them (TEKNIK_REHBER.md Sec. 4:
    'Ortam damgasi').
    """
    import torchvision

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "project_commit": get_git_commit_hash(project_root),
    }
