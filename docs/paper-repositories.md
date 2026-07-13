# Open Repositories for Thesis References

> This file lists the **open-source (public)** code repositories for papers cited in the
> thesis report. Each row is matched to a bibliography number ([1]–[23]) from the thesis.
> Purpose: quickly locate reference implementations for baseline reproduction and components
> to be overridden.
>
> **Status labels:**
> - ✅ **Official repo available** — can be cloned and used directly.
> - ⚠️ **No official repo found** — requires reimplementation.
> - 🔁 **Community reimplementation** — not official but a reliable copy exists.
>
> Last verified: 2026-07-13.

---

## 1. Core Repos Used Directly (high priority)

| # | Paper (short) | Repo | Status | Note |
|---|---------------|------|--------|------|
| [1] | **VOneNet** — Dapello et al. 2020, NeurIPS | https://github.com/dicarlolab/vonenet | ✅ | **Cloned** into this project (`./external/vonenet/`, per `technical-guide.md` `external/` convention — auto-cloned by `00_setup_and_data.ipynb`). V1 front-end + ResNet/CORnet/AlexNet back-end. Weights download from S3. |
| [2] | **EVNets** — Piper et al. 2025, NeurIPS | https://github.com/lucaspiper99/evnet | ✅ | Next-generation bio-inspired front-end (VOneBlock-based). arXiv: 2506.03089. Literature reference. |
| [3] | **FOVEA** — Thavamani et al. 2021, ICCV | https://github.com/tchittesh/fovea | ✅ | Saliency-driven differentiable KDE-warp magnification. Detection (Faster R-CNN/RetinaNet). |
| [6] | **CORnet-S** — Kubilius et al. 2019, NeurIPS | https://github.com/dicarlolab/CORnet | ✅ | Recurrent ventral stream reference. Also: https://github.com/dicarlolab/neurips2019 |
| [7] / [23] | **Brain-Score** — Schrimpf et al. | https://github.com/brain-score/vision | ✅ | V1/V2/V4/IT + behavioral benchmark suite. Full score via brain-score.org submission. |
| [11] | **LCL-V2Net** — Parthasarathy et al. 2024 | https://github.com/nikparth/LCL-V2 | ✅ | Layerwise complexity-matched SSL, V1/V2 front-end. arXiv: 2312.11436 |
| [12] | **Model Metamers** — Feather et al. 2023, Nat. Neuro. | https://github.com/jenellefeather/model_metamers_pytorch | ✅ | Metamer generation + recognizability. Older (2019): https://github.com/jenellefeather/model_metamers |
| [15] | **GFNet (Glance-and-Focus)** — Wang et al. 2020, NeurIPS | https://github.com/blackfeather-wang/GFNet-Pytorch | ✅ | Confidence-threshold halting + glance/focus. Reference for IT-feedback loop. |
| [17] | **NeVA** — Schwinn et al. 2022 | https://github.com/SchwinnL/NeVA | ✅ | Task-driven differentiable foveated attention / scanpath. arXiv: 2211.12100 / 2204.09093 |
| [19] | **NeuroFovea** — Deza et al. 2019, ICLR | https://github.com/ArturoDeza/NeuroFovea | ✅ | Foveated style-transfer metamer generation. PyTorch: https://github.com/ArturoDeza/NeuroFovea_PyTorch |
| [20]/[21] | **MFTMA** — Chung/Cohen et al. | https://github.com/schung039/neural_manifolds_replicaMFT | ✅ | Manifold capacity/radius/dimension certification (center-correlation included). |
| [22] | **Neural Population Geometry** — Dapello, Chung et al. 2021, NeurIPS | https://github.com/chung-neuroai-lab/adversarial-manifolds | ✅ | VOneResNet18 + MFTMA example notebook (`MFTMA_analyze_adversarial_representations.ipynb`). arXiv: 2111.06979 |

---

## 2. Papers Without Official Repos — Require Reimplementation

These components are **critical** to the thesis but no official open code has been
published — override/reimplementation is required (see `technical-guide.md` "Override
strategy"). This overlaps with the thesis's original contribution axis (trace-based
periphery, IT feedback).

| # | Paper | Status | Source / Action |
|---|-------|--------|-----------------|
| [9] | **R-Blur** — Shah & Raj 2023, NeurIPS | ⚠️ No public repo found | arXiv: 2308.00854. Foveal blur + saturation reduction algorithm to be **reimplemented** from the paper (differentiable transform). Core of thesis Phase 2. |
| [4] | **FoveaTer** — Jonnalagadda et al. 2022 | ⚠️ No official repo | arXiv: 2105.14173. Pooling-region foveated ViT; confidence-threshold fixation. Reimplement or override on top of timm-ViT. |
| [10] | **Blur-trained CNN** — Jang & Tong 2024, Nat. Comm. | ⚠️ Code link unverified | DOI: 10.1038/s41467-024-45679-0. Gaussian blur curriculum; reproduction possible with standard torchvision augmentation. |
| [8] | **BLT** — Kietzmann et al. 2019, PNAS | ⚠️ Official repo unverified | DOI: 10.1073/pnas.1905544116. Bottom-up/lateral/top-down recurrence; functionally replaced with CORnet. |
| [23] | **Watson (2014)** — mRGC RF density formula, *J. Vision* 14(7):15 | ⚠️ No repo needed (closed-form formula) | DOI: 10.1167/14.7.15. **Biological basis for the pooling scale** of trace-based periphery: $60\,s(r)\approx0.53+0.434\,r$. Coded directly into `foveation.py`. See `trace-based-noise-guide.md`. |
| [5] | **Lukanov et al.** 2021, Front. Comput. Neurosci. | ⚠️ No repo | Foveal-peripheral efficiency model; conceptual reference. |
| [13] | **Gizdov et al.** 2025, CVPR | ⚠️ No repo found | Foveation in large multimodal models; conceptual reference. |
| [14] | **Liao & Poggio** 2016 | ⚠️ No repo (CBMM Memo 047) | arXiv: 1604.03640. ResNet ↔ unrolled recurrence equivalence — theoretical justification, no code needed. |
| [18] | **Friston et al.** 2016 | ⚠️ No repo | Active inference / visual foraging — conceptual reference (IT feedback motivation). |
| [16] | **RAM** — Mnih et al. 2014, NeurIPS | 🔁 No official, strong reimpl available | Recommended: https://github.com/kevinzakka/recurrent-visual-attention (REINFORCE hard-attention reference). |

---

## 3. Batch Clone Commands (optional)

To pull all core repos in one step in Colab or a local environment:

```bash
# Core (used directly)
git clone --depth 1 https://github.com/dicarlolab/vonenet.git          # [1] (already present)
git clone --depth 1 https://github.com/tchittesh/fovea.git             # [3]
git clone --depth 1 https://github.com/dicarlolab/CORnet.git           # [6]
git clone --depth 1 https://github.com/brain-score/vision.git brain-score-vision  # [7]/[23]
git clone --depth 1 https://github.com/nikparth/LCL-V2.git             # [11]
git clone --depth 1 https://github.com/jenellefeather/model_metamers_pytorch.git # [12]
git clone --depth 1 https://github.com/blackfeather-wang/GFNet-Pytorch.git        # [15]
git clone --depth 1 https://github.com/SchwinnL/NeVA.git               # [17]
git clone --depth 1 https://github.com/ArturoDeza/NeuroFovea_PyTorch.git          # [19]
git clone --depth 1 https://github.com/schung039/neural_manifolds_replicaMFT.git  # [20]/[21]
git clone --depth 1 https://github.com/chung-neuroai-lab/adversarial-manifolds.git # [22]

# Reimplementation reference
git clone --depth 1 https://github.com/kevinzakka/recurrent-visual-attention.git  # [16] (community)
```

> **Note:** In Colab, each repo is added to `sys.path` (not installed via `pip install -e .`),
> and only the needed classes are **overridden** (see `technical-guide.md` §3). This ensures
> experiment notebooks remain intact even if the upstream API changes.
