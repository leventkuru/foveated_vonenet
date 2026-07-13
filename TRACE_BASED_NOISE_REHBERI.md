# Trace-Based Metameric Peripheral Noise — Yapı ve SNR Rehberi

> **Tezle bağ:** Bu rehber, tezin **özgün katkısı** olan trace-based metameric peripheral
> noise mekanizmasını (docx §7 madde 4, §8.3, §8.4) uygulanabilir bir matematiksel
> çerçeveye oturtur. Merkezî fikir: peripheral bozulmayı keyfi bir "blur şiddeti"
> parametresiyle değil, **sinyal-gürültü oranı (SNR)** ile kontrol etmek ve bu SNR
> profilini **Watson (2014)** insan mRGC anatomisine bağlamaktır.
>
> Paralel kaynak: `Biyolojik_Goru_Modelleri_Tez_Raporu.docx` (güncel sürüm; yazar:
> Resul Levent Kuru). Bkz. `DENEY_PLANI_ve_ABLATIONS.md` (C2 ekseni, A1–A5 + SNR taraması)
> ve `TEKNIK_REHBER.md §3` (override). Referans: Watson, A. B. (2014), *J. Vision* 14(7):15 [23].

---

## 1. Kavram: "trace", "metameric" ve neden gürültü?

İnsan peripheral vision'ı sinyali silmez; onun **istatistiksel bir izini (trace)** tutar.
İki nokta:

- **Metamer:** İnsan çevresel görüşünde, aynı yerel özet istatistiklere (texture statistics)
  sahip iki farklı görüntü **ayırt edilemez** — bunlara *metamer* denir. Yani periphery,
  sinyalin tam kopyasını değil, yerel istatistiklerini korur.
- **Trace-based noise:** Periphery'yi düz blur (yalnızca yüksek frekansı siler) veya
  yapısız i.i.d. noise (istatistiği yok sayar) ile değil, **altta yatan sinyalin yerel
  özet istatistiklerine eşlenmiş** (signal-conditioned) bir gürültü örneğiyle değiştiririz.
  Bu örnek, istatistikleri (trace) korurken faz/ince yapıyı rastgeleleştirir.

Bu üç rejimin farkı, tek bir büyüklükte özetlenebilir: **korunan sinyalin gücünün,
eklenen/rastgeleleştirilen bileşenin gücüne oranı** — yani **SNR**. Bu rehberin çekirdeği,
SNR'ı eksantrisiteye ve Watson mRGC ölçeğine bağlayan formülasyondur.

---

## 2. Watson (2014) ile pooling ölçeğinin biyolojik temeli

Fixation $\mathbf p_0$'a olan eksantrisite $r=\lVert \mathbf p-\mathbf p_0\rVert$ (derece).
Watson (2014), Curcio & Allen (1990) ganglion yoğunluğu ölçümlerinden midget RGC
receptive-field yoğunluğunun eksantrisiteyle azalışını verir:

$$
d_{gf}(r) = d_{gf}(0)\,\Big[a\,\big(1+\tfrac{r}{r_2}\big)^{-2} + (1-a)\,e^{-r/r_e}\Big]
$$

ve buradan **ortalama mRGC aralığının (spacing)** eksantrisiteyle yaklaşık lineer arttığını
türetir (docx §8.3):

$$
\boxed{\,60\,s(r) \approx 0.53 + 0.434\,r\,}\qquad\Longrightarrow\qquad
s(r) = \frac{0.53 + 0.434\,r}{60}\ \text{[derece]}
$$

Burada $s(r)$ ortalama mRGC aralığı; **pooling penceresi yarıçapı** buna orantılı seçilir:

$$
w(r) = \kappa\, s(r),\qquad \kappa\gtrsim 1\ \text{(pooling faktörü, hiperparametre)}.
$$

- $r\to 0$: $s(0)=0.53/60\approx 0.0088^\circ$ (foveada ~0.5 arcmin, kon aralığına yakın) →
  pencere ~1 piksel, sinyal **birebir** korunur.
- $r$ büyüdükçe $w(r)$ lineer büyür → daha geniş pencere → daha çok istatistiksel özetleme.

> Kritik nokta: pooling ölçeği **ölçülmüş insan anatomisinden** gelir, keyfi bir blur-σ
> takviminden değil. Bu, mekanizmayı nicel bir biyolojik temele oturtur (docx §7, §8.3).

---

## 3. SNR formalizasyonu (rehberin çekirdeği)

### 3.1 Peripheral temsili sinyal + koşullu gürültü olarak yazma

Pooling penceresi $\Omega(r)$ üzerinde yerel özet istatistik operatörü $S_{\Omega}(\cdot)$
olsun (Bölüm 5'te somutlaşır). Trace-based peripheral örnek $\hat I$, sinyalin istatistiğini
korur ama gerisini rastgeleleştirir:

$$
\hat I \sim P\big(\,\cdot \mid S_{\Omega(r)}(I)\,\big).
$$

Bunu additif ayrıştırmayla yazalım:

$$
\hat I(\mathbf p) = \underbrace{\bar I_{\Omega(r)}(\mathbf p)}_{\text{korunan sinyal (trace)}}
\;+\; \underbrace{n(\mathbf p)}_{\text{sinyale-koşullu metameric gürültü}},
$$

burada $\bar I_{\Omega}$ pencere içi düşük-değişkenli bileşen (korunan yapı), $n$ ise yerel
texture istatistiğine uyan ama fazı rastgele olan artıktır.

### 3.2 Eksantrisiteye bağlı yerel SNR

Yerel SNR'ı, korunan sinyal gücünün metameric gürültü gücüne oranı olarak tanımlarız:

$$
\mathrm{SNR}(r) \;=\; \frac{P_{\text{signal}}(r)}{P_{\text{noise}}(r)}
\;=\; \frac{\mathrm{Var}\big[\bar I_{\Omega(r)}\big]}{\mathbb E\big[\lVert I-\hat I\rVert^2_{\Omega(r)}\big]},
\qquad
\mathrm{SNR}_{\text{dB}}(r) = 10\log_{10}\mathrm{SNR}(r).
$$

**Watson ile bağ (temel ilişki).** Pooling **alanı** $A(r)\propto w(r)^2\propto s(r)^2$ ile
büyür. Pencere içinde $\sim A(r)$ örnek üzerinden özetleme yapıldığında ince-yapı
sadakati alanla ters orantılı azalır. Buradan SNR'ın eksantrisiteyle **kapanış (roll-off)**
yasası:

$$
\boxed{\;\mathrm{SNR}(r) \;=\; \mathrm{SNR}_0\left(\frac{s(0)}{s(r)}\right)^{\beta}
\;=\; \mathrm{SNR}_0\left(\frac{0.53}{0.53+0.434\,r}\right)^{\beta}\;}
$$

- $\beta=2$: alan-tabanlı özetleme (varsayılan, savunulabilir seçim).
- $\beta$ **ablation knob'u**: peripheral sadakat kapanışının dikliğini kontrol eder.
- $\mathrm{SNR}_0$: foveadaki (referans) sadakat; genelde çok yüksek (fovea keskin).

$r\to 0$'da $\mathrm{SNR}\to\mathrm{SNR}_0$ (sinyal korunur); $r$ büyüdükçe monoton düşer.

### 3.3 SNR'ı sentez parametresine bağlama (uygulanabilir knob)

Pratikte SNR'ı, **içerik (sinyal) ile texture-örneği (gürültü) arasında interpolasyon**
katsayısı $\alpha(r)\in[0,1]$ ile kontrol ederiz (NeuroFovea/AdaIN tarzı):

$$
\hat I_{\alpha}(\mathbf p) = \alpha(r)\,I(\mathbf p) + \big(1-\alpha(r)\big)\,T\big(S_{\Omega(r)}(I)\big),
$$

burada $T(\cdot)$ istatistiğe eşlenmiş texture sentezleyicidir (Bölüm 5). Küçük bir modelle,
$\alpha$ ile SNR arasındaki ilişki $\mathrm{SNR}(\alpha)\approx \dfrac{\alpha^2}{(1-\alpha)^2}\rho$
biçimindedir ($\rho$: içerik/texture güç oranı). Böylece hedef $\mathrm{SNR}(r)$ profilinden
$\alpha(r)$ geri çözülür:

$$
\alpha(r) = \frac{\sqrt{\mathrm{SNR}(r)/\rho}}{1+\sqrt{\mathrm{SNR}(r)/\rho}}.
$$

Sonuç: **Watson $s(r)$ → hedef $\mathrm{SNR}(r)$ → interpolasyon $\alpha(r)$** zinciriyle,
peripheral bozulma tamamen SNR üzerinden ve biyolojik temelle sürülür.

---

## 4. Üç periphery rejiminin SNR ile birleşik görünümü

Tezin A1–A5 ablation'ı (bkz. `DENEY_PLANI §3.1`), aslında **aynı SNR eksenindeki üç farklı
gürültü yapısıdır**:

| Rejim | Matematiksel yapı | SNR davranışı | Trace korunur mu? |
|-------|-------------------|---------------|-------------------|
| **Düz blur (R-Blur)** | Alçak-geçiren: $\hat I = I * G_{\sigma(r)}$, eklenen gürültü yok | Sinyal *zayıflatılır*, ayrık gürültü terimi yok; yüksek-frekans SNR $\to 0$ | Kısmen (sadece düşük frekans) |
| **i.i.d. noise** | $\hat I = I + \eta,\ \eta\sim\mathcal N(0,\sigma^2)$, **sinyalden bağımsız** | SNR ayarlanabilir ama gürültü texture'a koşullu değil | **Hayır** (istatistik eşlenmez) |
| **Trace-based metameric** | $\hat I \sim P(\cdot\mid S_{\Omega(r)}(I))$, $\mathrm{SNR}(r)$ Watson $s(r)$'den | $\mathrm{SNR}(r)=\mathrm{SNR}_0\,(s_0/s(r))^\beta$, **koşullu** | **Evet** (yerel texture istatistiği) |

Bu tablo, üç rejimi **karşılaştırılabilir** kılar: hepsini aynı ortalama SNR bütçesinde
eşleyip, farkın **gürültünün yapısından** (koşullu vs değil, texture-eşlenmiş vs değil)
geldiğini izole ederiz. Tezin iddiası: eşit SNR'da bile trace-based, insan hizalanması ve
robustness'ta diğer ikisini geçer.

---

## 5. Sentez algoritması (yerel istatistik → metameric örnek)

**Özet istatistik $S_\Omega$ seçenekleri (artan sadakat/maliyet):**
1. **AdaIN (mean/std):** $S=\{\mu_\Omega, \sigma_\Omega\}$; $\mathrm{AdaIN}(x,y)=\sigma(y)\frac{x-\mu(x)}{\sigma(x)}+\mu(y)$. En ucuz.
2. **VGG Gram matrisi:** $G^{\ell}_{ij}=\sum_k F^{\ell}_{ik}F^{\ell}_{jk}$ (Gatys texture kodu).
3. **Portilla–Simoncelli:** steerable pyramid marjinal + çapraz korelasyon istatistikleri. En biyolojik.

**Sentez (iki yol):**
- **Optimizasyon tabanlı (referans, yavaş):** $\displaystyle \min_{\hat I}\sum_{\text{bölge}} \big\lVert S_\Omega(\hat I)-S_\Omega(I)\big\rVert^2$ (Portilla–Simoncelli / Gatys).
- **Amortize (üretim, hızlı):** NeuroFovea/SideEye tarzı küçük ileri-besleme generatif ağ ($T$).
  **Zorunlu:** sentez ucuz olmalı; aksi halde foveation'ın verimlilik kazancını yer (docx §8.3).

**Pipeline (özet):**
```
1. r(p)  = eksantrisite haritası (fixation'dan)
2. w(p)  = κ·s(r) = κ·(0.53 + 0.434·r)/60           # Watson pooling ölçeği
3. S     = eksantrisite-değişken pencerelerde S_Ω(I) # yerel texture istatistiği
4. T     = S'ye eşlenmiş metameric örnek (AdaIN/Gram/PS)
5. α(r)  = hedef SNR(r) profilinden (Bölüm 3.3)
6. Îper  = α(r)·I + (1-α(r))·T
7. comp  = m_fovea·I_sharp + (1-m_fovea)·Îper        # yumuşak geçişli kompozit
```

---

## 6. SNR'ın tez iddialarıyla bağı (neden bu formülasyon değerli)

- **Robustness:** Düşük peripheral SNR, adversarial perturbation'ın çevredeki enerjisini
  metameric gürültüyle "boğar" → robustness ↑. Ama çok düşük SNR accuracy'yi düşürür →
  **SNR-accuracy-robustness üçgeninde bir tatlı nokta** vardır. $\beta$ ve $\mathrm{SNR}_0$
  taramasıyla bu nokta bulunur.
- **İnsan hizalanması:** $\mathrm{SNR}(r)$ profilini Watson mRGC'ye eşlemek, üretilen
  peripheral görüntüleri insan için **metamerik** (ayırt edilemez) kılmayı hedefler →
  metamer recognizability testinde (Feather [11]) düşük insan-ayırt-edilebilirliği beklenir.
- **MFTMA (sertifikasyon):** $\mathrm{SNR}(r)$ profili object manifold geometrisini etkiler.
  Hipotez: Watson-temelli SNR profili, birim compute başına **manifold capacity'yi**
  ($\alpha_c$) maksimize eder; düz blur ve i.i.d. noise'a göre daha düşük radius/dimension verir
  (bkz. `DENEY_PLANI §6`, Dapello–Chung [21]).

---

## 7. Ablation ve ölçüme bağlanış

Bu rehber, `DENEY_PLANI_ve_ABLATIONS.md`'deki **C2 ekseni** ve **A-grubu** ile şöyle eşleşir:

- **A2–A4:** üç rejim, **eşit ortalama SNR bütçesinde** karşılaştırılır (yapı etkisini izole eder).
- **A5 → SNR-profil taraması (yeni):** trace-based rejimde $\beta\in\{1,2,3\}$ ve
  $\mathrm{SNR}_0$ süpürülür; "Watson-temelli $\mathrm{SNR}(r)$" vs "düz (flat) SNR" vs
  "keyfi blur takvimi" karşılaştırılır.
- **Ölçülen SNR metriği:** her koşulda gerçekleşen $\mathrm{SNR}(r)$ eğrisi rapor edilir
  (eksantrisite-binlenmiş), böylece accuracy/robustness/capacity **SNR'ın fonksiyonu olarak**
  çizilir. Bu, tezin nicel ana grafiklerinden biridir.

---

## 8. Uygulama notları (override ile)

`TEKNIK_REHBER.md §3` uyarınca bu mekanizma, mevcut kütüphaneleri değiştirmeden **yeni bir
`nn.Module` (`Periphery`)** olarak eklenir ve `FoveatedModel` wrapper'ına takılır:

- `src/foveation.py` içinde: `WatsonPooling(r)`, `SNRProfile(beta, snr0)`, `TraceBasedPeriphery(stat='adain'|'gram'|'ps', synth='amortized')`.
- Rejim seçimi tek parametreyle: `periphery_mode ∈ {'blur','iid','trace'}` → A2/A3/A4.
- Differentiable tutulur (curriculum ve gerekirse uçtan-uca eğitim için); amortize sentez ağı
  ayrı ön-eğitilebilir.
- Stokastik olduğu için: değerlendirmede seed sabitle; **adversarial'da EOT-N zorunlu**
  (metameric örnek her ileri geçişte değişir).

---

## 9. Riskler ve azaltımlar

| Risk | Azaltım |
|------|---------|
| Sentez maliyeti foveation kazancını yer | Amortize küçük generatif ağ (NeuroFovea/SideEye); optimizasyon yalnızca referans/doğrulama için |
| $\alpha$–SNR ilişkisi modele bağlı sapabilir | $\rho$'yu ampirik kalibre et (ölçülen SNR eğrisini hedefe fit et) |
| Çok düşük peripheral SNR → accuracy düşer | SNR-accuracy tatlı noktasını $\beta,\mathrm{SNR}_0$ taramasıyla bul |
| EOT'suz adversarial → sahte robustness | Tüm adversarial ölçümlerde EOT-N (N≥10) |
| Watson formülü foveal ppd/görüş açısına bağlı | Görüntü→derece dönüşümünü (visual_degrees, ppd) VOneNet ile tutarlı sabitle |
