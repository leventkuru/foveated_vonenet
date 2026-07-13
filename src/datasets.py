"""Dataset preparation and normalization utilities shared by all notebooks.

Two input-normalization conventions coexist in this project (see notebook
00_setup_and_data.ipynb, "Math core: normalization conventions"):

  * VOneNet family (VOneResNet-50, VOneAlexNet, VOneCORnet-S) was trained with
    mean = std = (0.5, 0.5, 0.5), i.e. x' = 2x - 1 mapping [0, 1] -> [-1, 1].
  * Plain torchvision backbones (ResNet-50, AlexNet, CORnet-S baseline) use the
    standard ImageNet channel statistics.

Picking the wrong convention for a given model silently degrades accuracy without
raising an error, so `build_transform` takes an explicit `normalization` argument
instead of a single hard-coded default.
"""

import os
import tarfile
import zipfile

import requests
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

VONENET_MEAN = (0.5, 0.5, 0.5)
VONENET_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RAW_MEAN = (0.0, 0.0, 0.0)
RAW_STD = (1.0, 1.0, 1.0)

# The 15 corruption types defined by Hendrycks & Dietterich (2019), used throughout
# DENEY_PLANI_ve_ABLATIONS.md Sec. 4/5 (ImageNet-C / mCE).
CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
SEVERITY_LEVELS = (1, 2, 3, 4, 5)


def normalization_stats(normalization: str):
    if normalization == "vonenet":
        return VONENET_MEAN, VONENET_STD
    if normalization == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if normalization == "raw":
        return RAW_MEAN, RAW_STD
    raise ValueError(f"Unknown normalization convention: {normalization!r}")


def build_transform(image_size: int = 224, normalization: str = "imagenet", train: bool = True):
    """Build the torchvision transform pipeline for a given model family.

    Args:
        image_size: spatial resolution expected by the backbone (224 for the
            ResNet/AlexNet/CORnet-S/VOneNet family used throughout this project).
        normalization: "imagenet" for plain torchvision backbones (baselines B0, B1,
            B3), "vonenet" for the VOneNet family (baseline B2 and its ablations), or
            "raw" (identity mean=0/std=1, i.e. leaves images in native [0, 1] space).
            "raw" is used for adversarial-attack pipelines, which perturb pixels in
            true [0, 1] space and apply the model-specific normalization *inside* the
            model via `models.wrap_for_raw_pixel_input` -- this keeps epsilon in PGD
            comparable across models that use different normalization conventions.
        train: if True, applies RandomResizedCrop + horizontal flip augmentation;
            if False, applies a deterministic Resize + CenterCrop (evaluation).
    """
    mean, std = normalization_stats(normalization)
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_cifar_transform(normalization: str = "imagenet", train: bool = True):
    """CIFAR-10 variant: no resize (native 32x32), light augmentation only."""
    mean, std = normalization_stats(normalization)
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


def get_cifar10(data_dir: str, train: bool = True, normalization: str = "imagenet",
                 download: bool = True, smoke_test: bool = False):
    """CIFAR-10 with its canonical 50k/10k train/test split (torchvision-managed).

    Used for fast ablation iteration before scaling to ImageNet, per
    DENEY_PLANI_ve_ABLATIONS.md Sec. 4 ("Metodolojik kural").

    The official mirror (cs.toronto.edu) is occasionally slow or rate-limited, which
    can stall a "quick end-to-end check" for a long time. To keep the same honest
    fallback policy used for ImageNet:
      * smoke_test=False -> always attempts the real download (the correct behavior
        for an actual run; ~170 MB, one-time).
      * smoke_test=True  -> uses the real, already-cached copy if present, otherwise
        skips the download entirely and falls back to SyntheticImageFolder (pipeline
        check only), rather than blocking on a possibly slow/stalled connection.
    """
    transform = build_cifar_transform(normalization=normalization, train=train)
    root = os.path.join(data_dir, "cifar10")
    if smoke_test:
        try:
            return torchvision.datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
        except RuntimeError:
            print(f"[smoke_test] CIFAR-10 not found under {root}; using SyntheticImageFolder "
                  "instead of downloading (set CFG['smoke_test'] = False for a real run).")
            return SyntheticImageFolder(num_samples=32, num_classes=10, image_size=32)
    return torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)


class SyntheticImageFolder(Dataset):
    """Shape-correct random-tensor stand-in for an ImageNet-scale dataset.

    Used ONLY when CFG['smoke_test'] is True and the real dataset is not present
    locally, so the data pipeline (and everything downstream of it) can be exercised
    end-to-end without a multi-GB download. Every sample is deterministic given its
    index (seeded per-item), but this is NOT real data: any accuracy computed on it is
    meaningless by construction and must never be reported as a result.
    """

    def __init__(self, num_samples: int = 32, num_classes: int = 10, image_size: int = 224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        image = torch.rand(3, self.image_size, self.image_size, generator=g)
        label = int(torch.randint(0, self.num_classes, (1,), generator=g).item())
        return image, label


def get_imagenet(data_dir: str, split: str = "val", normalization: str = "imagenet",
                  image_size: int = 224, smoke_test: bool = True, num_classes: int = 1000):
    """ImageNet-1K loader.

    Expects the canonical ImageFolder layout at `data_dir/imagenet/<split>/<class>/*.JPEG`.
    Full ImageNet-1K (~150 GB) is not downloaded automatically -- mount it from Google
    Drive, or a Kaggle/Hugging-Face-backed copy, and point `data_dir` at it. If the
    directory is missing:
      * smoke_test=True  -> falls back to SyntheticImageFolder (pipeline check only)
      * smoke_test=False -> raises FileNotFoundError naming the expected path, per the
        "never fabricate results" rule.
    """
    root = os.path.join(data_dir, "imagenet", split)
    if os.path.isdir(root):
        transform = build_transform(image_size=image_size, normalization=normalization, train=(split == "train"))
        return torchvision.datasets.ImageFolder(root=root, transform=transform)
    if smoke_test:
        print(f"[smoke_test] ImageNet-1K not found at {root}; using SyntheticImageFolder instead.")
        return SyntheticImageFolder(num_samples=32, num_classes=num_classes, image_size=image_size)
    raise FileNotFoundError(
        f"ImageNet-1K '{split}' split not found at {root}. Mount it from Google Drive or "
        "download via Kaggle/Hugging Face (requires accepting the ILSVRC license) and "
        "place it in this layout, or set CFG['smoke_test'] = True to run the pipeline "
        "on synthetic data."
    )


def get_imagenet100(data_dir: str, split: str = "val", normalization: str = "imagenet",
                     image_size: int = 224, smoke_test: bool = True):
    """ImageNet-100 loader (100-class subset used for fast, FoveaTer-style efficiency
    comparisons, per DENEY_PLANI_ve_ABLATIONS.md Sec. 4). Same fallback policy as
    `get_imagenet`."""
    root = os.path.join(data_dir, "imagenet100", split)
    if os.path.isdir(root):
        transform = build_transform(image_size=image_size, normalization=normalization, train=(split == "train"))
        return torchvision.datasets.ImageFolder(root=root, transform=transform)
    if smoke_test:
        print(f"[smoke_test] ImageNet-100 not found at {root}; using SyntheticImageFolder instead.")
        return SyntheticImageFolder(num_samples=32, num_classes=100, image_size=image_size)
    raise FileNotFoundError(
        f"ImageNet-100 '{split}' split not found at {root}. Place a 100-class "
        "ImageFolder-layout subset there, or set CFG['smoke_test'] = True to run the "
        "pipeline on synthetic data."
    )


def download_file(url: str, dst_path: str, chunk_size: int = 1 << 20) -> bool:
    """Streamed download with a clear failure mode: returns False (and prints a
    warning) instead of raising, so a stale or unreachable URL degrades gracefully
    rather than crashing the whole notebook."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        return True
    except Exception as exc:
        print(f"[download_file] Failed to download {url}: {exc}")
        return False


def _extract_archive(path: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    if path.endswith(".tar") or path.endswith(".tar.gz") or path.endswith(".tgz"):
        with tarfile.open(path) as tf:
            tf.extractall(dst_dir)
    elif path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dst_dir)
    else:
        raise ValueError(f"Unrecognized archive format: {path}")


def prepare_benchmark_dataset(name: str, data_dir: str, urls=None, instructions: str = "") -> bool:
    """Generic preparer for the corruption/robustness benchmark sets (ImageNet-C,
    ImageNet-R, ImageNet-Sketch, Stylized-ImageNet).

    These are distributed from a handful of stable academic hosts (see
    https://github.com/hendrycks/robustness for ImageNet-C/R, and the Stylized-ImageNet
    / ImageNet-Sketch project pages for the others), but exact file URLs occasionally
    move between Zenodo/Drive/Kaggle hosts. Rather than hard-coding a link that could go
    stale and fail silently, this function:
      * downloads + extracts every URL in `urls` if supplied (a single string, or a
        list -- ImageNet-C in particular ships as four separate category tars: blur,
        digital, noise, weather, all extracted into the same `dst_dir`). Verify each
        URL against the source page above before running a full, non-smoke-test
        experiment.
      * otherwise prints the expected directory layout and returns False, so the
        calling notebook can skip this dataset without crashing.

    Each downloaded file keeps the extension from its URL (required for
    `_extract_archive` to detect tar vs. zip); a URL with no recognizable archive
    extension is rejected up front rather than silently mis-extracted.
    """
    dst_dir = os.path.join(data_dir, name)
    if os.path.isdir(dst_dir) and os.listdir(dst_dir):
        print(f"{name}: already present at {dst_dir}")
        return True

    url_list = [urls] if isinstance(urls, str) else (urls or [])
    if url_list:
        ok = True
        for url in url_list:
            basename = os.path.basename(url.split("?")[0])
            if not basename.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                print(f"{name}: skipping unrecognized archive extension in URL: {url}")
                ok = False
                continue
            archive_path = os.path.join(data_dir, f"{name}_{basename}")
            print(f"{name}: downloading {basename} from {url} ...")
            if download_file(url, archive_path):
                _extract_archive(archive_path, dst_dir)
                os.remove(archive_path)
            else:
                ok = False
        if ok:
            return True

    print(
        f"{name}: not found at {dst_dir} and no working URL was provided or a download failed.\n"
        f"  Verify the current download link at https://github.com/hendrycks/robustness "
        f"(or the dataset's project page) and either:\n"
        f"    (a) pass it as urls=... (string or list of strings), or\n"
        f"    (b) manually place the extracted data at {dst_dir}\n"
        f"  {instructions}"
    )
    return False
