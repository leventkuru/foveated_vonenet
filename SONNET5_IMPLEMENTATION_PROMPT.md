# Sonnet 5 için Uygulama Prompt'u — Foveated Vision Tezi Notebook'ları

> **Kullanım:** Aşağıdaki tüm metni (bu başlık dahil) Sonnet 5 modeline tek bir görev
> promptu olarak verin. Model, çalışma dizinindeki dosyalara erişebiliyor olmalı
> (`Biyolojik_Goru_Modelleri_Tez_Raporu.docx` [esas] ve `.pdf`, `TEKNIK_REHBER.md`,
> `DENEY_PLANI_ve_ABLATIONS.md`, `MAKALE_REPOLARI.md`, `TRACE_BASED_NOISE_REHBERI.md`,
> `vonenet/`).

---

## ROL VE HEDEF

Sen, biyolojik olarak ilham alınmış görü modelleri üzerine bir yüksek lisans tezinin
deneysel altyapısını kuran kıdemli bir araştırma mühendisisin. Görevin, aşağıda tanımlı
**7 Jupyter notebook'u (`.ipynb`)** baştan sona, çalıştırılabilir ve **matematiksel olarak
gerekçelendirilmiş** biçimde üretmek. Bu, tezin önerdiği mimarinin (attention-güdümlü
foveation + V1 front-end + recurrent ventral stream + IT-feedback loop; özgün katkı:
trace-based metameric peripheral noise) referans implementasyonudur.

## ÖNCE OKU (bağlam dosyaları)

Kod yazmadan önce şu dosyaları oku ve içselleştir:

1. `Biyolojik_Goru_Modelleri_Tez_Raporu.docx` — **tezin güncel/esas sürümü** (mimari §8,
   biyolojik eşleme §9, deney planı §10). Çelişkide bu belge kazanır. (`.pdf` daha eski bir
   kopyadır; yalnızca genel bağlam için.)
2. `TEKNIK_REHBER.md` — Colab kurulumu, notebook organizasyonu ve **override stratejisi**
   (kütüphaneleri yeniden yazma; subclass / monkey-patch / wrapper ile değiştir).
3. `DENEY_PLANI_ve_ABLATIONS.md` — bileşenler (C1–C5), baseline'lar (B0–B5), ablation
   matrisi (A/B/D/E grupları), metrikler ve MFTMA metodolojisi.
4. `MAKALE_REPOLARI.md` — kullanılacak resmi repolar ve reimplement edilecek makaleler.
5. `TRACE_BASED_NOISE_REHBERI.md` — **özgün katkının** (trace-based metameric periphery)
   matematiksel çerçevesi: SNR formalizasyonu + Watson (2014) mRGC temeli. Notebook 02'nin
   matematik çekirdeği birebir buna dayanır.
6. `vonenet/vonenet/` — VOneBlock'un gerçek implementasyonu (`modules.py`, `vonenet.py`).
   Override'ları buna göre yaz.

> Not: Güncel tezde (docx) subcortical/EVNets bloğu **yoktur**, Watson (2014) [23] mRGC
> temeli **vardır**. Eski PDF ile çelişirse docx'i izle.

## BAĞLAYICI KURALLAR (ihlal etme)

1. **Ortam Google Colab.** Her notebook `TEKNIK_REHBER.md §1.1`'deki standart açılış
   hücresiyle başlar (Drive mount, `IN_COLAB` tespiti, `PROJECT_ROOT`, checkpoint dizini).
   Sürümleri sabitle (pinned pip). Uzun eğitimlerde **checkpoint + resume** zorunlu.
2. **Her deliverable bir `.ipynb`.** Gerçek notebook JSON'u üret (nbformat v4). Ortak
   yardımcılar `src/*.py`'ye gider ama **deney mantığı ve override'lar notebook
   hücrelerinde görünür** olmalı.
3. **Yeniden yazma — override et.** Repolar `external/`'a klonlanır, `sys.path`'e eklenir,
   **hiç düzenlenmez**. Davranış değişiklikleri subclass/monkey-patch/wrapper ile yapılır
   (bkz. `TEKNIK_REHBER.md §3`). Her monkey-patch'in geri-alma yolu olsun.
4. **Matematiksel arka plan zorunlu.** Her notebook, kod hücrelerinden ÖNCE, kullandığı
   yöntemin matematiğini **Markdown + LaTeX** ile açıklayan hücreler içerir (aşağıdaki
   "MATEMATİK ÇEKİRDEĞİ" listelerine birebir uy). Formüller `$...$` / `$$...$$` ile;
   her sembol tanımlanır; kod, yazılan denklemi birebir uygular ve yorumla denkleme atıf yapar.
5. **Tekrarlanabilirlik.** Sabit seed (torch/numpy/random + cudnn deterministic),
   `CFG` dict, sonuçları `results/NN_*.json` ve şekilleri `results/NN_*.png` olarak yaz.
   VOneBlock stokastik olduğu için değerlendirmede `vone_block.fix_noise(seed=...)` kullan.
6. **Stokastik model = EOT zorunlu.** Adversarial değerlendirmede (PGD/APGD) stokastik
   front-end/periphery için gradyanı **EOT-N (N≥10)** ile al; aksi halde robustluk yapay olur.
7. **Önce küçük ölçek.** Her yöntem önce CIFAR-10'da doğrulanır, sonra ImageNet(-100/-1K)'e
   taşınır. Notebook'lar bir `DATASET` bayrağıyla ikisini de desteklesin.
8. **Dürüstlük.** Bir şey çalışmıyorsa veya atlandıysa notebook'ta açıkça belirt (TODO/UYARI
   hücresi). Uydurma sonuç/sözde-metrik yazma.
9. **Tüm çıktı İngilizce.** Üretilen **her kod ve parça İngilizce** yazılır: kod, değişken/fonksiyon
   isimleri, yorumlar (comments), docstring'ler, Markdown açıklama hücreleri (matematik dahil),
   `print`/log mesajları, şekil başlıkları ve eksen etiketleri, `README.md`. (Yalnızca bu planlama
   dokümanları — `TEKNIK_REHBER.md`, `DENEY_PLANI_ve_ABLATIONS.md` vb. — Türkçe kalır; onlara
   atıfta bulunabilirsin ama notebook içeriğini İngilizce üret.)

## ÜRETİLECEK NOTEBOOK'LAR VE MATEMATİK ÇEKİRDEĞİ

Sırayla üret. Her biri kendi kendine çalışabilir olmalı ama bir öncekinin çıktısını okuyabilir.

### `notebooks/00_setup_and_data.ipynb`
- İçerik: Drive mount, repo klonlama (`MAKALE_REPOLARI.md §3`), veri hazırlığı (CIFAR-10,
  ImageNet-100/1K, ImageNet-C/R/Sketch, Stylized-ImageNet), normalizasyon
  (VOneNet: mean=std=0.5), `src/` iskeleti.
- **MATEMATİK ÇEKİRDEĞİ:** veri normalizasyonu, train/val bölme; ImageNet-C şiddet
  seviyeleri ve **mCE** tanımı: $\mathrm{CE}_c = \frac{\sum_s E_{s,c}}{\sum_s E_{s,c}^{\text{AlexNet}}}$,
  $\mathrm{mCE}=\frac{1}{15}\sum_c \mathrm{CE}_c$.

### `notebooks/01_baseline_reproduce.ipynb`  (Baseline'lar B0–B3)
- İçerik: ResNet-50, AlexNet, VOneResNet-50, CORnet-S'i yükle/eğit; bir robustness
  diliminde (ImageNet-C veya PGD) baseline kazancını doğrula.
- **MATEMATİK ÇEKİRDEĞİ (VOneBlock — LNP modeli):**
  - Gabor filtresi: $g(x,y)=\exp\!\big(-\frac{x'^2}{2\sigma_x^2}-\frac{y'^2}{2\sigma_y^2}\big)\cos(2\pi f x' + \phi)$,
    $x'=x\cos\theta+y\sin\theta$. Simple/complex hücre kanalları.
  - Simple cell: $r_s=[\,k\,(g_{q0}*I)\,]_+$ (yarı-dalga doğrultma). Complex cell (quadrature):
    $r_c=\frac{k}{\sqrt2}\sqrt{(g_{q0}*I)^2+(g_{q1}*I)^2}$.
  - Nöral (Poisson-benzeri) stokastisite: $\tilde r = r + \xi\sqrt{\mathrm{ReLU}(r)+\epsilon}$,
    $\xi\sim\mathcal N(0,1)$ (bkz. `modules.py: noise_f`, neuronal mod). `k_exc`, `noise_scale`,
    `noise_level` parametreleri.
  - PGD saldırısı: $x_{t+1}=\Pi_{\|\delta\|_\infty\le\varepsilon}\big(x_t+\alpha\,\mathrm{sign}\nabla_x \ell\big)$.
  - Not: Liao & Poggio (2016) — ResNet residual bloklarının **unrolled recurrence**'a denkliği
    (mimari gerekçe olarak açıkla).

### `notebooks/02_foveation_rblur_and_periphery.ipynb`  (Ablation A1–A6, E1–E3) — ÖZGÜN ÇEKİRDEK
- İçerik: R-Blur foveal transform (reimplement, [9]), üç periphery rejimi
  (düz blur / i.i.d. noise / **trace-based metameric**), blur-şiddeti curriculum,
  differentiable foveal warp (FOVEA [3] referans).
- **ZORUNLU GÖRSELLEŞTİRME (trace-based noise sanity-check):** Herhangi bir eğitimden ÖNCE,
  veri kümelerinden birkaç örnek al (CIFAR-10 ve ImageNet'ten ≥6 sample) ve trace-based noise
  eklenmiş halini görselleştir. Her sample için yan yana bir panel çiz:
  `[orijinal] · [R-Blur foveal] · [düz blur çevre] · [i.i.d. noise çevre] · [trace-based metameric çevre]`,
  fixation noktası işaretli ve üstünde ölçülen $\mathrm{SNR}(r)$ değeri yazılı. Ek olarak
  eksantrisiteye karşı $\mathrm{SNR}(r)$ eğrisini ve $\beta$ taramasının etkisini çiz.
  Şekilleri `results/02_periphery_samples.png` ve `results/02_snr_profile.png` olarak kaydet.
  Amaç: noise'un görüntüyü *ne yaptığını* sayısal metriklerden önce gözle doğrulamak.
- **MATEMATİK ÇEKİRDEĞİ:**
  - Eksantrisite-bağımlı acuity: fixation $\mathbf p_0$'a uzaklık $r=\|\mathbf p-\mathbf p_0\|$;
    blur ölçeği $\sigma(r)=\sigma_0(1+\kappa r)$ (lineer) veya kortikal magnifikasyon
    $M(r)\propto 1/(r+r_0)$. Çok-ölçekli Gauss piramidi ile mekânsal olarak değişen konvolüsyon.
  - Renk satürasyonu düşürme: $s(r)=\max(0,\,1-\lambda r)$, HSV/S kanalına uygulanır.
  - **Trace-based metameric periphery (özgün katkı) — SNR temelli.** Bu bölümün matematiği
    `TRACE_BASED_NOISE_REHBERI.md` ile birebir aynı olmalı:
    - **Watson (2014) pooling ölçeği:** $60\,s(r)\approx 0.53+0.434\,r$, pencere $w(r)=\kappa\,s(r)$.
    - **Yerel özet istatistik** $S_\Omega$: AdaIN ($\mu,\sigma$) / VGG Gram $G_{ij}^\ell=\sum_k F_{ik}^\ell F_{jk}^\ell$ / Portilla–Simoncelli.
    - **Metamer sentezi:** $\hat I\sim P(\cdot\mid S_{\Omega(r)}(I))$; optimizasyon
      $\min_{\hat I}\sum_{\text{bölge}}\|S_\Omega(\hat I)-S_\Omega(I)\|^2$ veya amortize ağ.
    - **SNR formalizasyonu (çekirdek):** $\mathrm{SNR}(r)=\dfrac{P_{\text{signal}}(r)}{P_{\text{noise}}(r)}
      =\mathrm{SNR}_0\big(s_0/s(r)\big)^{\beta}$; $\mathrm{SNR}_{\text{dB}}=10\log_{10}\mathrm{SNR}$.
    - **SNR knob'u:** içerik–texture interpolasyonu $\hat I_\alpha=\alpha(r)\,I+(1-\alpha(r))\,T(S_\Omega(I))$,
      $\alpha(r)=\dfrac{\sqrt{\mathrm{SNR}(r)/\rho}}{1+\sqrt{\mathrm{SNR}(r)/\rho}}$.
  - **Üç rejimi SNR ekseninde birleştir:** düz blur (alçak-geçiren, eklenen gürültü yok) /
    i.i.d. noise ($\eta\sim\mathcal N$, sinyalden bağımsız — VOneNet Poisson'u gibi) /
    trace-based (sinyale koşullu, texture-eşlenmiş). A2–A4 **eşit ortalama SNR bütçesinde**
    karşılaştırılır; fark gürültünün *yapısından* gelir. **Gerçekleşen $\mathrm{SNR}(r)$ eğrisini
    ölç ve raporla** (ana grafik: metrik vs SNR).
  - Kompozit görüntü: keskin fovea $\odot$ maske + metameric çevre $\odot$ (1−maske).

### `notebooks/03_v1_block.ipynb`  (Ablation B7–B9, B12–B13)
- İçerik: V1 front-end ablation'ı (yok / VOneBlock noise-off / Poisson-on) `noise_f`
  override ile; ventral back-end değişimi (ResNet-50 ↔ CORnet-S).
- **MATEMATİK ÇEKİRDEĞİ:**
  - VOneBlock LNP (yukarıdakine atıf) + noise on/off ablation'ının gradyan akışına etkisi.
  - **Brain-Score predictivity:** model aktivasyonu $\to$ nöral yanıt için PLS/lineer regresyon;
    açıklanan varyans $R^2$, gürültü tavanına (noise ceiling) normalize edilmiş skor.
  - CORnet-S recurrence: $h_t = f(W_{ff}x + W_{rec}h_{t-1})$; zaman-açılımı (unrolling) ve
    IT dinamikleri.

### `notebooks/04_it_feedback_multiglance.ipynb`  (Ablation D1–D4)
- İçerik: attention/saliency → fixation seçimi (top-k/NMS) → multi-glance loop →
  confidence/uncertainty ile halting; differentiable (soft) vs REINFORCE (hard) varyantları
  (GFNet [15], RAM [16], NeVA [17] referans). `FoveatedModel` wrapper (`TEKNIK_REHBER.md §3.3`).
- **MATEMATİK ÇEKİRDEĞİ:**
  - Saliency/attention rollout: self-attention katmanları $\tilde A=\prod_\ell (A_\ell+I)$;
    fixation için top-k / NMS.
  - Belirsizlik: softmax $p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$; **entropi** $H(p)=-\sum_i p_i\log p_i$;
    confidence $=\max_i p_i$. Halting koşulu: $\max_i p_i \ge \tau$ olunca dur.
  - **REINFORCE (hard attention):** politika $\pi_\phi(l\mid s)$; hedef $J=\mathbb E[R]$;
    gradyan $\nabla_\phi J=\mathbb E\big[R\,\nabla_\phi\log\pi_\phi(l\mid s)\big]$; varyans için
    baseline $b$: $(R-b)\nabla_\phi\log\pi_\phi$. Ödül: doğru sınıflandırma.
  - **Differentiable (soft) yol:** task loss tüm glance'lerden geri yayılır; fixation için
    yumuşak (softmax-ağırlıklı) örnekleme / Gumbel-softmax.
  - **Counterfactual loss:** bir konuma bakmanın doğru-sınıf olasılığını artırıp artırmadığını
    ödüllendiren ek terim.
  - Adaptive computation / evidence accumulation (ACC analojisi): beklenen kazanç eşiği.

### `notebooks/05_mftma_certification.ipynb`  (Tüm koşullar × MFTMA + metamer)
- İçerik: her ablation koşulunun seçili katmanlarından (V1 çıkışı, ventral orta, IT readout)
  sınıf-manifold'larını çıkar; `neural_manifolds_replicaMFT` ([20]/[21]) ile sertifikala;
  metamer recognizability ([12]); adversarial-manifolds ([22]) ile kıyas.
- **MATEMATİK ÇEKİRDEĞİ (replica mean-field manifold analizi):**
  - $P$ manifold'un $N$-boyutta lineer ayrılabilirliği; yük $\alpha=P/N$; **manifold capacity**
    $\alpha_c$ (rastgele dichotomy'ler altında ayrılabilirliğin kritik yükü).
  - **Manifold radius** $R_M$ ve **manifold dimension** $D_M$ (anchor noktaları ve Gaussian
    istatistiklerinden); **center correlation**. İlişki: düşük $R_M, D_M$ ve düşük merkez
    korelasyonu $\Rightarrow$ yüksek $\alpha_c$.
  - Hipotez testi: foveation + V1 noise + trace-based periphery + IT feedback eklendikçe
    $\alpha_c\uparrow$, $R_M,D_M\downarrow$ mı? (davranışsal + geometrik iki imza).

### `notebooks/06_full_model_and_ablations.ipynb`
- İçerik: tam kapalı-loop model (C1=R-Blur, C2=trace-based, C3=VOneBlock Poisson,
  C4=ResNet-50, C5=multi-glance+halting); tüm ablation matrisini koştur; **leave-one-out**
  özet tablosu (`DENEY_PLANI_ve_ABLATIONS.md §8`); sonuç şekilleri/tabloları.
- **MATEMATİK ÇEKİRDEĞİ:** marjinal katkı = full skoru − (bileşen çıkarılmış) skoru;
  verimlilik eğrisi (accuracy vs ortalama glance / GFLOP); EOT-N ortalaması
  $\bar g=\frac1N\sum_{i}\nabla_x \ell(f(T_i(x)),y)$ ile adversarial gradyan.

## ÇALIŞMA BİÇİMİ

- Notebook'ları **numaralı sırayla** üret; her birini üretmeden önce ilgili repo/kod
  bölümlerini incele (varsayım yapma, `vonenet` kodunu ve klonladığın repoları oku).
- Her notebook'un başına kısa bir "Amaç / Bu notebook'taki matematik / Ablation ID'leri /
  Girdi-Çıktı" özet hücresi koy.
- Ağır eğitim adımlarını `CFG['smoke_test']=True` ile 1-2 iterasyonda çalışabilir yap
  (Colab'de hızlı doğrulama için), tam eğitim ayrı bayrağa bağlı olsun.
- Ortak kod tekrarını `src/`'ye taşı: `overrides.py`, `foveation.py`, `it_feedback.py`,
  `mftma.py`, `eval_harness.py`.
- İşin sonunda kısa bir `notebooks/README.md` ile çalıştırma sırasını ve her notebook'un
  ürettiği çıktıları belgele.

## KABUL KRİTERLERİ

- [ ] 7 notebook + `src/` modülleri + `notebooks/README.md` üretildi.
- [ ] **Tüm notebook içeriği İngilizce** (kod, yorumlar, docstring'ler, Markdown/matematik
      hücreleri, print/log, şekil etiketleri, README).
- [ ] Her notebook Colab açılış hücresi, `CFG`, seed, checkpoint/resume içeriyor.
- [ ] Her notebook, yukarıdaki "MATEMATİK ÇEKİRDEĞİ" maddelerini LaTeX'li Markdown
      hücrelerinde açıklıyor ve kod bu denklemleri birebir uyguluyor.
- [ ] Notebook 02, eğitimden önce **trace-based noise görselleştirmesini** üretiyor
      (örnek panelleri + $\mathrm{SNR}(r)$ eğrisi, `results/`'a kaydedilmiş).
- [ ] Hiçbir upstream repo düzenlenmedi; tüm değişiklikler override.
- [ ] Adversarial değerlendirmeler EOT kullanıyor.
- [ ] `smoke_test` modunda tüm notebook'lar hatasız baştan sona çalışıyor.

**İlk adım:** Bağlam dosyalarını oku, `src/` iskeletini ve
`notebooks/00_setup_and_data.ipynb`'i üret, sonra onay için özetini sun.
