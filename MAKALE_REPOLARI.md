# Tez Kaynakçasındaki Makalelerin Açık Repository'leri

> Bu dosya, tez raporunda (`Biyolojik_Goru_Modelleri_Tez_Raporu.pdf`) atıf yapılan
> makalelerin **açık kaynak (public)** kod depolarını listeler. Her satır, tezdeki
> kaynakça numarasına ([1]–[23]) göre eşleştirilmiştir. Amaç: baseline reprodüksiyonu
> ve override edilecek referans implementasyonlarını hızlıca bulmak.
>
> **Durum etiketleri:**
> - ✅ **Resmi repo mevcut** — doğrudan klonlanıp kullanılabilir.
> - ⚠️ **Resmi repo yok / bulunamadı** — yeniden implementasyon (reimplementation) gerekir.
> - 🔁 **Topluluk reimplementasyonu** — resmi değil ama güvenilir kopya var.
>
> Son doğrulama: 2026-07-13.

---

## 1. Doğrudan Kullanılacak Çekirdek Repolar (yüksek öncelik)

| # | Makale (kısa) | Repo | Durum | Not |
|---|---------------|------|-------|-----|
| [1] | **VOneNet** — Dapello vd. 2020, NeurIPS | https://github.com/dicarlolab/vonenet | ✅ | Bu projeye **klonlandı** (`./external/vonenet/`, `TEKNIK_REHBER.md`'deki `external/` konvansiyonuyla — `00_setup_and_data.ipynb` tarafından otomatik klonlanır). V1 front-end + ResNet/CORnet/AlexNet back-end. Ağırlıklar S3'ten iniyor. |
| [2] | **EVNets** — Piper vd. 2025, NeurIPS | https://github.com/lucaspiper99/evnet | ✅ | Yeni nesil biyo-esinli front-end (VOneBlock tabanlı). arXiv: 2506.03089. Literatür referansı. |
| [3] | **FOVEA** — Thavamani vd. 2021, ICCV | https://github.com/tchittesh/fovea | ✅ | Saliency güdümlü differentiable KDE-warp magnification. Detection (Faster R-CNN/RetinaNet). |
| [6] | **CORnet-S** — Kubilius vd. 2019, NeurIPS | https://github.com/dicarlolab/CORnet | ✅ | Recurrent ventral stream referansı. Ek: https://github.com/dicarlolab/neurips2019 |
| [7] / [23] | **Brain-Score** — Schrimpf vd. | https://github.com/brain-score/vision | ✅ | V1/V2/V4/IT + davranış benchmark suite. Tam skor için brain-score.org submission. |
| [11] | **LCL-V2Net** — Parthasarathy vd. 2024 | https://github.com/nikparth/LCL-V2 | ✅ | Layerwise complexity-matched SSL, V1/V2 front-end. arXiv: 2312.11436 |
| [12] | **Model Metamers** — Feather vd. 2023, Nat. Neuro. | https://github.com/jenellefeather/model_metamers_pytorch | ✅ | Metamer üretimi + recognizability. Eski (2019): https://github.com/jenellefeather/model_metamers |
| [15] | **GFNet (Glance-and-Focus)** — Wang vd. 2020, NeurIPS | https://github.com/blackfeather-wang/GFNet-Pytorch | ✅ | Confidence-eşikli halting + glance/focus. IT-feedback loop için referans. |
| [17] | **NeVA** — Schwinn vd. 2022 | https://github.com/SchwinnL/NeVA | ✅ | Task-driven differentiable foveated attention / scanpath. arXiv: 2211.12100 / 2204.09093 |
| [19] | **NeuroFovea** — Deza vd. 2019, ICLR | https://github.com/ArturoDeza/NeuroFovea | ✅ | Foveated style-transfer metamer üretimi. PyTorch: https://github.com/ArturoDeza/NeuroFovea_PyTorch |
| [20]/[21] | **MFTMA** — Chung/Cohen vd. | https://github.com/schung039/neural_manifolds_replicaMFT | ✅ | Manifold capacity/radius/dimension sertifikasyonu (center-correlation dahil). |
| [22] | **Neural Population Geometry** — Dapello, Chung vd. 2021, NeurIPS | https://github.com/chung-neuroai-lab/adversarial-manifolds | ✅ | VOneResNet18 + MFTMA örnek notebook'u (`MFTMA_analyze_adversarial_representations.ipynb`). arXiv: 2111.06979 |

---

## 2. Resmi Repo Bulunamayan / Yeniden Yazılacak Makaleler

Bu bileşenler tez için **kritik** ama resmi açık kod yayınlanmamış — override/reimplementation
gerektirir (bkz. `TEKNIK_REHBER.md` "Override stratejisi"). Bu, tezin özgün katkı ekseniyle
(trace-based periphery, IT feedback) örtüşür.

| # | Makale | Durum | Kaynak / Aksiyon |
|---|--------|-------|------------------|
| [9] | **R-Blur** — Shah & Raj 2023, NeurIPS | ⚠️ Public repo bulunamadı | arXiv: 2308.00854. Foveal blur + saturasyon düşürme algoritması makaleden **reimplement** edilecek (differentiable transform). Tezin Aşama 2 çekirdeği. |
| [4] | **FoveaTer** — Jonnalagadda vd. 2022 | ⚠️ Resmi repo yok | arXiv: 2105.14173. Pooling-region tabanlı foveated ViT; confidence-eşikli fixation. Reimplement veya timm-ViT üzerine override. |
| [10] | **Blur-trained CNN** — Jang & Tong 2024, Nat. Comm. | ⚠️ Kod linki doğrulanamadı | DOI: 10.1038/s41467-024-45679-0. Gaussian blur curriculum; standart torchvision augmentation ile reprodüksiyon mümkün. |
| [8] | **BLT** — Kietzmann vd. 2019, PNAS | ⚠️ Resmi repo doğrulanamadı | DOI: 10.1073/pnas.1905544116. Bottom-up/lateral/top-down recurrence; CORnet ile fonksiyonel ikame edilir. |
| [23] | **Watson (2014)** — mRGC RF yoğunluk formülü, *J. Vision* 14(7):15 | ⚠️ Repo gerekmez (kapalı-form formül) | DOI: 10.1167/14.7.15. Trace-based periphery'nin **pooling ölçeğinin biyolojik temeli**: $60\,s(r)\approx0.53+0.434\,r$. Doğrudan `foveation.py`'ye kodlanır. Bkz. `TRACE_BASED_NOISE_REHBERI.md`. |
| [5] | **Lukanov vd.** 2021, Front. Comput. Neurosci. | ⚠️ Repo yok | Foveal-peripheral verimlilik modeli; kavramsal referans. |
| [13] | **Gizdov vd.** 2025, CVPR | ⚠️ Repo bulunamadı | Foveation in large multimodal models; kavramsal referans. |
| [14] | **Liao & Poggio** 2016 | ⚠️ Repo yok (CBMM Memo 047) | arXiv: 1604.03640. ResNet↔unrolled recurrence denkliği — teorik gerekçe, kod gerektirmez. |
| [18] | **Friston vd.** 2016 | ⚠️ Repo yok | Active inference / visual foraging — kavramsal referans (IT feedback motivasyonu). |
| [16] | **RAM** — Mnih vd. 2014, NeurIPS | 🔁 Resmi yok, güçlü reimpl var | Önerilen: https://github.com/kevinzakka/recurrent-visual-attention (REINFORCE'lu hard-attention referansı). |

---

## 3. Toplu Klonlama Komutları (opsiyonel)

Colab veya lokal ortamda çekirdek repoları tek seferde çekmek için:

```bash
# Çekirdek (doğrudan kullanılacak)
git clone --depth 1 https://github.com/dicarlolab/vonenet.git          # [1] (zaten mevcut)
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

# Reimplementasyon referansı
git clone --depth 1 https://github.com/kevinzakka/recurrent-visual-attention.git  # [16] (topluluk)
```

> **Not:** Colab'de her repo `pip install -e .` yerine `sys.path.append(...)` ile eklenip,
> ihtiyaç duyulan sınıflar **override** edilecek (bkz. `TEKNIK_REHBER.md` §3). Böylece
> upstream API değişse bile deney notebook'ları kırılmaz.
