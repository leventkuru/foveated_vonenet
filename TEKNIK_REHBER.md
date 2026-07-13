# Teknik Rehber — Deney Altyapısı, Colab Kurulumu ve Override Stratejisi

> **Tez:** Biyolojik Olarak İlham Alınmış Görü Modelleri — Attention-güdümlü foveation,
> V1 front-end (VOneNet), recurrent ventral stream ve IT-feedback loop.
> **Bu belgenin amacı:** Deney ortamını (Colab), kod organizasyonunu ve mevcut
> kütüphaneleri **override** ederek nasıl ilerleneceğini tanımlamak.

---

## 0. Temel İlkeler (bağlayıcı kurallar)

Bu tezin tüm deneysel altyapısı üç ilke etrafında kurulur:

1. **Çalışma ortamı Google Colab'dir.** Tüm eğitim/değerlendirme deneyleri Colab (tercihen
   Colab Pro, A100/L4 GPU) üzerinde koşacak şekilde tasarlanır. Uzun süren ImageNet
   eğitimleri için checkpoint'ler Google Drive'a yazılır; oturum kopmalarına karşı
   **resume-from-checkpoint** her notebook'ta zorunludur.

2. **Tüm kodlar notebook (`.ipynb`) formatındadır.** Her deney, kendi kendine yeten,
   baştan sona çalıştırılabilir bir Jupyter notebook'tur. `.py` modülleri yalnızca
   ortak yardımcı fonksiyonlar için kullanılır ve notebook içinden import edilir; ama
   **deneyin mantığı ve override'lar notebook hücrelerinde açıkça görünür** olmalıdır
   (reprodüksiyon şeffaflığı için).

3. **Mevcut kütüphaneler yeniden yazılmaz — override edilir.** VOneNet, CORnet,
   FOVEA vb. depolar klonlanır ve `sys.path`'e eklenir. İhtiyaç duyulan davranış
   değişiklikleri (örn. V1 noise'un kapatılması, foveal transform'un öne eklenmesi,
   IT-feedback loop) **orijinal sınıflar override/monkey-patch edilerek** yapılır.
   Upstream kod hiç fork'lanıp elle düzenlenmez; böylece kaynağa sadakat korunur ve
   ablation'lar tek bir bayrakla açılıp kapanabilir.

---

## 1. Colab Çalışma Ortamı

### 1.1 Standart açılış hücresi (her notebook'un başında)

```python
# --- Ortam tespiti: Colab mı, lokal mi? ---
import sys, os, subprocess
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = '/content/drive/MyDrive/tez_foveated_vision'
else:
    PROJECT_ROOT = os.path.expanduser('~/Desktop/Bogazici/Tez/foveated_vision')

os.makedirs(PROJECT_ROOT, exist_ok=True)
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)

# GPU kontrolü
import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

### 1.2 Bağımlılıklar

Her notebook, sürüm-sabitli (pinned) bir kurulum hücresi içerir. **Sürümleri sabitleyin**
(Colab imajı değiştikçe kırılmayı önler):

```python
!pip -q install torch==2.* torchvision timm==1.* foolbox==3.* robustbench \
    scipy scikit-learn matplotlib tqdm
# Brain-Score yalnızca değerlendirme notebook'unda:
# !pip -q install git+https://github.com/brain-score/vision.git
```

### 1.3 Repoların path'e eklenmesi (fork YOK)

```python
REPOS = {
    'vonenet': 'https://github.com/dicarlolab/vonenet.git',
    'CORnet':  'https://github.com/dicarlolab/CORnet.git',
    'fovea':   'https://github.com/tchittesh/fovea.git',
}
REPO_DIR = os.path.join(PROJECT_ROOT, 'external')
os.makedirs(REPO_DIR, exist_ok=True)
for name, url in REPOS.items():
    dst = os.path.join(REPO_DIR, name)
    if not os.path.exists(dst):
        subprocess.run(['git', 'clone', '--depth', '1', url, dst], check=True)
    if dst not in sys.path:
        sys.path.insert(0, dst)
```

> Depolar `data/` altına değil `external/` altına klonlanır ve **hiç düzenlenmez**.
> Bu, `MAKALE_REPOLARI.md`'deki resmi repolarla birebir aynı kalmayı garanti eder.

---

## 2. Depo / Notebook Organizasyonu

Google Drive üzerindeki `tez_foveated_vision/` kökü:

```
tez_foveated_vision/
├── external/                 # klonlanan resmi repolar (READ-ONLY, override edilir)
│   ├── vonenet/  CORnet/  fovea/ ...
├── src/                      # ortak yardımcı .py modülleri (notebook'lardan import)
│   ├── overrides.py          # tüm monkey-patch/override sınıfları burada toplanır
│   ├── foveation.py          # R-Blur, trace-based periphery transformları
│   ├── it_feedback.py        # confidence/uncertainty güdümlü fixation + halting
│   ├── mftma.py              # manifold capacity sertifikasyon sarmalayıcısı
│   └── eval_harness.py       # ImageNet-C, PGD/APGD, Brain-Score köprüleri
├── notebooks/                # TÜM deneyler .ipynb (numaralı, sıralı)
│   ├── 00_setup_and_data.ipynb
│   ├── 01_baseline_reproduce.ipynb
│   ├── 02_foveation_rblur_and_periphery.ipynb
│   ├── 03_v1_block.ipynb
│   ├── 04_it_feedback_multiglance.ipynb
│   ├── 05_mftma_certification.ipynb
│   └── 06_full_model_and_ablations.ipynb
├── checkpoints/              # model ağırlıkları (resume için)
└── results/                  # metrik CSV/JSON + şekiller
```

**Notebook adlandırma kuralı:** `NN_kısa_ad.ipynb`. Her notebook yalnızca kendi
aşamasından sorumludur ve çıktısını `results/` altına yazar; bir sonraki notebook onu okur.

---

## 3. Override Stratejisi (kütüphaneleri yeniden yazmadan değiştirme)

Deneylerin çoğu, mevcut sınıfların davranışını noktasal olarak değiştirmeyi gerektirir.
Bunun için **üç kalıp** kullanılır. Hepsi `src/overrides.py` içinde toplanır ve notebook'ta
tek satırla etkinleştirilir.

### 3.1 Kalıp A — Alt sınıflama (subclassing) ile davranış değiştirme

En temiz yöntem. Orijinal sınıfı bozmadan, gereken metodu ezersiniz.
Örnek: VOneBlock'un Poisson noise'unu ablation için kapatılabilir hale getirmek.

```python
# src/overrides.py
from vonenet.modules import VOneBlock

class ConfigurableVOneBlock(VOneBlock):
    """VOneBlock + noise'u runtime'da aç/kapa (ablation: V1 Poisson noise on/off)."""
    def __init__(self, *args, noise_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_enabled = noise_enabled

    def noise_f(self, x):
        if not self._noise_enabled:
            return self.noise(x)          # sadece ReLU, stokastik terim yok
        return super().noise_f(x)         # orijinal neuronal/gaussian noise
```

### 3.2 Kalıp B — Monkey-patch (kaynağa hiç dokunmadan çalışma zamanında değiştirme)

Sınıfı alt-sınıflamak pratik değilse (örn. derin bir back-end içinde), metodu
çalışma zamanında değiştiririz. Ablation bayrağıyla geri alınabilir olmalıdır.

```python
import vonenet.modules as vm
_orig_noise_f = vm.VOneBlock.noise_f      # orijinali sakla (geri almak için)

def _no_noise(self, x):
    return self.noise(x)

def set_v1_noise(enabled: bool):
    vm.VOneBlock.noise_f = _orig_noise_f if enabled else _no_noise
```

> **Kural:** Her monkey-patch'in bir "geri alma" (restore) yolu olmalı ki notebook
> içinde ablation koşulları arasında temiz geçiş yapılabilsin.

### 3.3 Kalıp C — Sarmalama (wrapper) ile yeni bileşen ekleme

Foveation, trace-based periphery ve IT-feedback gibi **tezin özgün katkıları** upstream'de
yoktur; bunlar mevcut modeli sarmalayan yeni `nn.Module`'lar olarak eklenir. VOneNet zaten
`nn.Sequential(vone_block, bottleneck, model_back_end)` yapısındadır (`vonenet/vonenet.py`);
biz bu sıranın **önüne** foveal transform, **etrafına** IT-feedback loop koyarız.

```python
import torch.nn as nn

class FoveatedModel(nn.Module):
    """Foveal transform + (VOneNet gövdesi) + IT-feedback loop — hepsi sarmalayıcı."""
    def __init__(self, core, foveate, periphery, it_head=None):
        super().__init__()
        self.foveate = foveate       # attention -> fixation -> R-Blur foveal warp
        self.periphery = periphery   # trace-based metameric noise (özgün katkı)
        self.core = core             # override edilmiş VOneNet/ResNet gövdesi
        self.it_head = it_head       # confidence/uncertainty + halting (opsiyonel)

    def forward(self, x, fixation=None, max_glances=1):
        logits = None
        for t in range(max_glances):
            fov = self.foveate(x, fixation)
            comp = self.periphery(x, fov, fixation)   # keskin fovea + metameric çevre
            logits, conf, unc = self._readout(comp)
            if self.it_head is None or self.it_head.should_halt(conf):
                break
            fixation = self.it_head.next_fixation(x, conf, unc)  # IT feedback
        return logits

    def _readout(self, comp):
        out = self.core(comp)
        if self.it_head is not None:
            return self.it_head(out)
        return out, None, None
```

> Not: `nn.DataParallel` sarmalaması (`get_model` içinde) nedeniyle, override'lardan
> önce `model.module` ile gerçek modele erişin; wrapper'ları `module` seviyesine takın.

### 3.4 Override haritası (hangi ablation → hangi override)

| Tez bileşeni / ablation | Hedef sınıf | Override kalıbı |
|-------------------------|-------------|-----------------|
| V1 Poisson noise on/off | `vonenet.modules.VOneBlock.noise_f` | A veya B |
| Foveation on/off (R-Blur) | girişe eklenen `Foveate` modülü | C (wrapper) |
| Periphery rejimi (blur / i.i.d. / trace-based) | `Periphery` modülü, strateji parametresi | C |
| IT feedback on/off, multi-glance | `FoveatedModel` loop + `it_head` | C |
| Back-end: ResNet-50 ↔ CORnet-S | `vonenet.vonenet.VOneNet(model_arch=...)` | mevcut parametre |
| Curriculum vs sabit blur | eğitim döngüsündeki blur zamanlayıcısı | notebook eğitim hücresi |

---

## 4. Tekrarlanabilirlik (reproducibility) Kuralları

Her notebook şunları içermelidir:

- **Sabit seed:** `torch.manual_seed`, `numpy.random.seed`, `random.seed` + cudnn deterministik.
  Not: VOneNet'in stokastik front-end'i için `vone_block.fix_noise(seed=...)` ile
  değerlendirmede noise sabitlenir (bkz. `modules.py: fix_noise/unfix_noise`).
- **Config bloğu:** tüm hiperparametreler tek bir `CFG` dict'inde; sonuç dosyasına
  JSON olarak yazılır.
- **Checkpoint + resume:** her epoch sonunda Drive'a kaydet; başta varsa yükle.
- **Sonuç kaydı:** metrikler `results/NN_*.json` + şekiller `results/NN_*.png`.
- **Ortam damgası:** `torch.__version__`, GPU adı, commit hash'leri sonuç JSON'ına eklenir.

---

## 5. Değerlendirme Köprüleri (eval harness)

`src/eval_harness.py` içinde toplanır, notebook'lardan çağrılır:

- **Temiz accuracy:** ImageNet-1K / CIFAR-10 val.
- **Corruption robustness:** ImageNet-C (15 corruption × 5 şiddet) → mCE; ImageNet-R/Sketch.
- **Adversarial:** PGD ve APGD (foolbox / robustbench); stokastik front-end için
  **EOT (Expectation over Transformation)** ile gradient — aksi halde noise sahte robustluk verir.
- **Shape bias / cue-conflict:** Geirhos stylized-ImageNet seti.
- **Brain-Score:** V1/V2/V4/IT predictivity; lokal public benchmark + gerekirse submission.
- **MFTMA:** `src/mftma.py` (bkz. `neural_manifolds_replicaMFT`) — capacity/radius/dimension.

> **Kritik uyarı (stokastik modeller):** VOneNet ve trace-based periphery stokastiktir.
> Adversarial değerlendirmede EOT kullanılmazsa robustluk yapay çıkar. Tüm adversarial
> ablation'lar EOT-N (N≥10) ile raporlanır.

---

## 6. Özet İş Akışı

1. `00_setup_and_data.ipynb` → Drive mount, repo klonla, veri kümelerini hazırla.
2. `01_baseline_reproduce.ipynb` → VOneResNet-50 + ResNet-50 baseline, 1 robustness diliminde doğrula.
3. `02..04` → foveation, V1 front-end, IT-feedback bileşenlerini **override ile** ekle.
4. `05_mftma_certification.ipynb` → geometrik sertifikasyon.
5. `06_full_model_and_ablations.ipynb` → tam kapalı-loop model + `DENEY_PLANI_ve_ABLATIONS.md`'deki tüm ablation'lar.

Ayrıntılı ablation matrisi ve metodolojiler için: **`DENEY_PLANI_ve_ABLATIONS.md`**.
Repo linkleri için: **`MAKALE_REPOLARI.md`**.
