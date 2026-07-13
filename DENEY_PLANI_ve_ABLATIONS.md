# Deney Planı ve Ablation Matrisi

> **Tez:** Attention-güdümlü foveation + V1 front-end + recurrent ventral
> stream + IT-feedback loop; **özgün katkı:** trace-based metameric peripheral noise.
> **Sertifikasyon:** MFTMA (manifold capacity/radius/dimension).
> Bu belge, tez raporunun 8., 9. ve 10. bölümlerini (önerilen mimari, biyolojik eşleme,
> deneysel plan) uygulanabilir bir **ablation matrisi + metodoloji** olarak açar.
>
> Uygulama kuralları (Colab, `.ipynb`, override) için: `TEKNIK_REHBER.md`.
> Repo linkleri için: `MAKALE_REPOLARI.md`.

---

## 1. Model Bileşenleri (ablation eksenleri)

Önerilen mimari 5 bağımsız açılıp-kapanabilir bileşene ayrılır. Her bileşen bir
**ablation ekseni**dir; her eksen tek bir override bayrağıyla kontrol edilir.

| # | Bileşen | Ablation değerleri | Override hedefi |
|---|---------|--------------------|-----------------|
| C1 | **Foveation (uzamsal acuity)** | kapalı / R-Blur foveal warp | girişe eklenen `Foveate` wrapper |
| C2 | **Periphery rejimi** | düz blur / i.i.d. noise / **trace-based metameric** (özgün, Watson-temelli SNR) | `Periphery` strateji parametresi |
| C3 | **V1 front-end (Gabor + noise)** | yok / VOneBlock (noise off) / VOneBlock (Poisson on) | `VOneBlock.noise_f` override |
| C4 | **Ventral back-end** | ResNet-50 (ana) / CORnet-S (recurrent referans) | `VOneNet(model_arch=...)` |
| C5 | **IT feedback + fixation** | tek-glance / multi-glance + confidence-halting | `FoveatedModel` loop + `it_head` |

**Tam (full) model:** C1=R-Blur, C2=trace-based, C3=VOneBlock(Poisson on),
C4=ResNet-50, C5=multi-glance+halting.

---

## 2. Baseline'lar (referans noktaları)

| Baseline | Amaç | Kaynak/override |
|----------|------|-----------------|
| **B0 · ResNet-50** | Ana feedforward omurga, ölçek referansı | torchvision |
| **B1 · AlexNet** | Klasik zayıf baseline (VOneNet karşılaştırması) | torchvision |
| **B2 · VOneResNet-50** | V1 front-end katkısının anchor'ı | `vonenet` [1] |
| **B3 · CORnet-S** | Recurrent ventral stream referansı (Brain-Score) | `CORnet` [6] |
| **B4 · R-Blur ResNet-50** | Foveal-training robustluk referansı | reimpl [9] |
| **B5 · LCL-V2Net** | Self-supervised V1/V2 front-end referansı | `LCL-V2` [11] |

> B0–B3 **reprodüksiyon zorunlu** (Aşama 1). B4–B5 ilgili ablation aşamasında eklenir.

---

## 3. Ablation Matrisi (deney tablosu)

Her satır bir deney koşulu. `→` sütunu, o ablation'ın **hangi bilimsel soruyu**
yalıtladığını gösterir. Öncelik: **P1** (çekirdek, mutlaka) > **P2** (önemli) > **P3** (varsa).

### 3.1 Foveation & Periphery (özgün katkı ekseni) — Notebook 02

| ID | C1 Foveation | C2 Periphery | → İzole edilen soru | Öncelik |
|----|-------------|-------------|---------------------|---------|
| A1 | kapalı | — | Foveation'sız taban (full-res) | P1 |
| A2 | R-Blur | düz blur | Klasik foveal blur katkısı | P1 |
| A3 | R-Blur | i.i.d. noise (sinyalden bağımsız) | Yapısız gürültü vs blur | P2 |
| A4 | R-Blur | **trace-based metameric** | **Özgün katkının net etkisi** | **P1** |
| A5 | R-Blur | trace-based (blur/fixation curriculum) | Curriculum'ın katkısı | P2 |
| A6 | R-Blur | trace-based, **SNR(r) profil taraması** | SNR kapanışı ve biyolojik temelin etkisi | P1 |

**A2–A4 karşılaştırma kuralı:** üç periphery rejimi **eşit ortalama SNR bütçesinde**
eşlenir; böylece fark, gürültünün *yapısından* (sinyale koşullu mu, texture-eşlenmiş mi)
gelir — SNR miktarından değil. Formülasyon: `TRACE_BASED_NOISE_REHBERI.md §3–4`.

**A6 (SNR taraması):** trace-based rejimde pooling ölçeği **Watson (2014) mRGC** aralığına
bağlıdır: $60\,s(r)\approx 0.53+0.434\,r$; SNR kapanışı
$\mathrm{SNR}(r)=\mathrm{SNR}_0\,(s_0/s(r))^{\beta}$. Süpürülenler: $\beta\in\{1,2,3\}$,
$\mathrm{SNR}_0$ seviyeleri; ve **"Watson-temelli $\mathrm{SNR}(r)$" vs "düz (flat) SNR"
vs "keyfi blur takvimi"**. Amaç: accuracy–robustness tatlı noktası + biyolojik temelin katkısı.

**Ölçülen:** temiz acc, ImageNet-C mCE, PGD/APGD (EOT), shape bias, GFLOP/latency,
**gerçekleşen $\mathrm{SNR}(r)$ eğrisi** (eksantrisite-binlenmiş). Ana grafik: metrikler
**SNR'ın fonksiyonu** olarak çizilir.
**Hipotez:** A4 > A3 ≳ A2 > A1 robustlukta (eşit SNR'da bile trace-based önde); Watson-temelli
$\mathrm{SNR}(r)$ (A6) insan peripheral texture temsiline en yakın — metamer recognizability
(düşük insan-ayırt-edilebilirliği) + MFTMA capacity ile doğrulanır.

### 3.2 V1 front-end — Notebook 03

| ID | C3 V1 | → İzole edilen soru | Öncelik |
|----|-------|---------------------|---------|
| B7 | yok | Front-end'siz taban | P1 |
| B8 | VOneBlock (noise off) | Gabor bank'ın tek başına katkısı | P1 |
| B9 | VOneBlock (Poisson on) | Stokastik noise'un robustluk katkısı | P1 |

**Ölçülen:** V1/V2 Brain-Score predictivity, adversarial robustness (EOT), shape bias.
**Not:** B9 için EOT-N zorunlu; noise gradient maskeleme yapmasın.

### 3.3 Ventral back-end & Recurrence — Notebook 03/04

| ID | C4 Back-end | → İzole edilen soru | Öncelik |
|----|------------|---------------------|---------|
| B12 | ResNet-50 | Feedforward taban (residual = unrolled recurrence, Liao & Poggio 2016) | P1 |
| B13 | CORnet-S | Açık recurrence'ın Brain-Score/robustluk katkısı | P2 |

**Ölçülen:** V4/IT Brain-Score, ImageNet acc, parametre/FLOP verimliliği.

### 3.4 IT Feedback & Multi-glance — Notebook 04

| ID | C5 loop | Halting | Eğitim sinyali | → İzole edilen soru | Öncelik |
|----|---------|---------|----------------|---------------------|---------|
| D1 | tek-glance | — | task loss | Feedback'siz taban | P1 |
| D2 | multi-glance | confidence-eşik | differentiable (soft) | Fixation tekrarının katkısı | P1 |
| D3 | multi-glance | confidence-eşik | REINFORCE (hard attention) | RL vs differentiable karşılaştırması | P2 |
| D4 | multi-glance | entropy/uncertainty-eşik | soft + counterfactual loss | Uncertainty-güdümlü fixation katkısı | P3 |

**Ölçülen:** accuracy vs ortalama glance sayısı (verimlilik eğrisi), adaptive-compute
kazancı, küçük/oklüzyonlu nesnelerde accuracy, latency.
**Referanslar:** GFNet [15] (confidence halting), RAM [16] (REINFORCE), NeVA [17] (foveated attention).

### 3.5 Eğitim rejimi ablation'ı — tüm notebook'lar

| ID | Rejim | → İzole edilen soru | Öncelik |
|----|-------|---------------------|---------|
| E1 | Sabit blur şiddeti | Taban | P2 |
| E2 | Blur/fixation **curriculum** | Curriculum'ın biyoloji-kodlama katkısı | P1 |
| E3 | Curriculum + adversarial-free | AT olmadan robustluk mümkün mü? | P1 |

---

## 4. Veri Kümeleri

| Küme | Kullanım | Aşama |
|------|----------|-------|
| **CIFAR-10** | Hızlı iterasyon / prototipleme (tüm ablation'lar önce burada) | 2–4 |
| **ImageNet-1K** | Ana temiz accuracy ve ölçek | 1–5 |
| **ImageNet-C** | Common corruption robustness (mCE) | 1–5 |
| **ImageNet-R / Sketch** | Domain shift, OOD | 3–5 |
| **Stylized-ImageNet (Geirhos)** | Shape bias / cue-conflict | 5 |
| **ImageNet-100** | FoveaTer tarzı verimlilik karşılaştırması | 2, 4 |
| **Brain-Score neural setleri** | V1/V2/V4/IT predictivity | 3–5 |

> **Metodolojik kural:** Her ablation önce CIFAR-10'da hızlı doğrulanır, sonra
> ImageNet ölçeğine taşınır (tez raporu §11 notu). IT-feedback ve trace-based periphery
> gibi özgün bileşenler önceliklendirilir.

---

## 5. Metrikler

| Kategori | Metrik | Araç |
|----------|--------|------|
| **Doğruluk** | top-1/top-5 temiz accuracy | standart |
| **Corruption** | mCE (ImageNet-C), ImageNet-R/Sketch acc | `eval_harness` |
| **Adversarial** | PGD, APGD robust acc — **EOT-N (N≥10)** stokastik modelde | foolbox/robustbench |
| **Beyin hizalama** | V1/V2/V4/IT predictivity | Brain-Score [7] |
| **İnsan hizalama** | shape bias, cue-conflict accuracy | Geirhos seti |
| **Algısal boşluk** | metamer recognizability | model_metamers_pytorch [12] |
| **Geometri (sertifikasyon)** | manifold capacity, radius, dimension, center-corr | MFTMA [20/21] |
| **Peripheral sadakat** | gerçekleşen **$\mathrm{SNR}(r)$ eğrisi** (eksantrisite-binlenmiş), Watson uyum hatası | `foveation.py` ölçüm |
| **Verimlilik** | GFLOP, latency, bellek, ortalama fixation sayısı | profiler |

> **SNR ana grafiği:** accuracy / mCE / adversarial acc / manifold capacity, her biri
> **$\mathrm{SNR}(r)$ profilinin fonksiyonu** olarak çizilir (üç periphery rejimi tek eksende).
> Formülasyon ve ölçüm: `TRACE_BASED_NOISE_REHBERI.md`.

---

## 6. MFTMA Sertifikasyon Metodolojisi

Accuracy ve Brain-Score'a **ek** olarak, her ablation'ın representational geometry'sini
MFTMA ile ölçüp geometrik bir imza raporlarız (tez raporu §10.1).

**Prosedür:**
1. Her koşul için, seçili katmanlardan (V1 çıkışı, ventral orta katman, IT readout)
   sınıf-başına aktivasyon manifold'ları çıkarılır (sınıf başına ~50 örnek).
2. `neural_manifolds_replicaMFT` ile **capacity (αc), radius (Rm), dimension (Dm),
   center-correlation** hesaplanır.
3. Katmanlar ve ablation koşulları boyunca karşılaştırılır.

**Doğrudan ölçülebilir hipotezler (tezden):**
- Foveation + V1 noise + trace-based periphery + IT feedback ekledikçe object
  manifold'ları **daha separable** olur mu? → **daha yüksek capacity, daha düşük
  radius/dimension** beklenir.
- Stokastik biyo-esinli front-end'lerin capacity artışı, noise'un robust algıdaki
  rolünü açıklar mı (Dapello, Chung vd. [22]) → adversarial-manifolds bulgusuyla kıyas.

Böylece her ablation **iki imza** ile raporlanır: davranışsal (accuracy/robustness) +
geometrik (capacity).

---

## 7. Deney → Notebook → Aşama Eşlemesi

| Notebook | Kapsadığı ablation'lar | Tez aşaması (raporun §11) |
|----------|------------------------|---------------------------|
| `00_setup_and_data` | — (veri + repo kurulumu) | Aşama 0 (13–19 Tem) |
| `01_baseline_reproduce` | B0–B3 | Aşama 1 (20 Tem–2 Ağu) |
| `02_foveation_rblur_and_periphery` | A1–A6, E1–E3 | Aşama 2 (3–16 Ağu) |
| `03_v1_block` | B7–B9, B12, B13 | Aşama 3 (17–30 Ağu) |
| `04_it_feedback_multiglance` | D1–D4 | Aşama 4 (31 Ağu–13 Eyl) |
| `05_mftma_certification` | tüm koşullar × MFTMA + metamer | Aşama 5 (14–23 Eyl) |
| `06_full_model_and_ablations` | Full model + leave-one-out özet | Aşama 6 (24–28 Eyl) |

---

## 8. Leave-One-Out Özet Deneyi (nihai)

Full modelden her bileşeni **tek tek çıkararak** (C1…C5) marjinal katkı ölçülür:

| Koşul | Çıkarılan bileşen | Beklenen düşüş |
|-------|-------------------|----------------|
| Full | — | referans (en yüksek) |
| −C1 | foveation | verimlilik ↓, küçük nesne acc ↓ |
| −C2 | trace-based → düz blur (eşit SNR) | insan hizalama ↓, metamer benzerliği ↓ |
| −C3 | V1 noise | adversarial robustness ↓ (EOT) |
| −C4 | ResNet→CORnet-S ile değiştir | Brain-Score ↑? , FLOP değişimi |
| −C5 | IT feedback | adaptive-compute kazancı ↓, zor örnek acc ↓ |

Bu tablo tezin ana katkı iddiasını (her bileşenin sinerjik katkısı) tek bakışta kanıtlar.

---

## 9. Riskler ve Azaltımlar

| Risk | Azaltım |
|------|---------|
| Colab oturum kopması / uzun ImageNet eğitimi | Checkpoint+resume; önce CIFAR-10; alt-küme (ImageNet-100) |
| Stokastik front-end'de sahte robustluk | Tüm adversarial ölçümlerde EOT-N zorunlu |
| R-Blur/FoveaTer resmi repo yok | Makaleden reimplement (bkz. `MAKALE_REPOLARI.md` §2) |
| Trace-based periphery sentez maliyeti | NeuroFovea/SideEye tarzı amortize edilmiş küçük generatif ağ [19] |
| Upstream API değişikliği | Fork yok; sadece override (bkz. `TEKNIK_REHBER.md` §3) |
| Tempo darlığı (plan Temmuz ortası başlıyor) | P1 ablation'lar önce; P3 ancak zaman kalırsa |
