# CNN Image Classification (Keras & OpenCV)

Bu repo, görüntü sınıflandırma alanında öğrenme amacıyla yaptığım iki küçük ama uçtan uca çalışan denemeyi içerir.  
Odak nokta: **basit ama derli toplu bir pipeline** kurmak.

- İkili sınıflandırma + OpenCV ile tek görsel üzerinde tahmin
- CIFAR-10 üzerinde çok sınıflı CNN + data augmentation

---

## 📁 Proje Yapısı

```text
.
├── 01-binary-custom-opencv/
│   └── binary_image_classification_pipeline.ipynb
├── 02-cifar10-cnn-augmentation/
│   └── cifar10_cnn.py
└── README.md
```

---

## 🔹 Proje 1 – Binary Image Classification + OpenCV

**Klasör:** `01-binary-custom-opencv/`  
**Tür:** İkili görüntü sınıflandırma (örnek: Cat vs Dog, ama herhangi iki sınıf olabilir)

Bu notebook şunları gösterir:

- Klasör yapısına göre (`training_set/`, `test_set/`) görüntü verisini yükleme
- Keras ile basit bir CNN modeli kurma ve eğitme
- Eğitim ve doğrulama accuracy/loss grafikleriyle süreci izleme
- OpenCV ile tek bir görüntüyü okuyup modele vererek tahmin alma
- Eğitilen modeli diske kaydetme (`.keras`)

### Çalıştırma Adımları (özet)

1. Dataset’i şu yapıda hazırlayın:

   ```text
   training_set/
       class0/
       class1/
   test_set/
       class0/
       class1/
   ```

2. Notebook içindeki yol ayarlarını kendi klasör yapınıza göre güncelleyin.
3. Notebook’u baştan sona çalıştırın.
4. Son kısımda, tekil bir görüntü yolu vererek model tahmini alın.

---

## 🔹 Proje 2 – CIFAR-10 CNN + Data Augmentation

**Klasör:** `02-cifar10-cnn-augmentation/`  
**Tür:** 10 sınıflı görüntü sınıflandırma (CIFAR-10)

Bu script şunları yapar:

- `cifar10` veri setini otomatik olarak indirir ve yükler
- Veriyi [0, 1] aralığına normalize eder
- Etiketleri one-hot encoding formatına çevirir
- `ImageDataGenerator` ile data augmentation uygular
- Basit bir CNN modeli kurup eğitir
- Eğitim / doğrulama accuracy ve loss grafiğini çizer
- Test seti için `classification_report` çıktısı üretir
- Eğitilen modeli `cifar10_cnn.keras` olarak kaydeder

### Çalıştırma

Klasöre girip:

```bash
python cifar10_cnn.py
```

Komutuyla modeli eğitebilirsiniz. Varsayılan ayarlar:

- Batch size: 64
- Epoch: 30
- Optimizer: RMSprop (learning_rate=1e-4, decay=1e-6)

---

## 🧩 Gereksinimler

Projeler için temel bağımlılıklar:

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

- Python 3.8+ önerilir.
- GPU varsa TensorFlow otomatik kullanır, yoksa CPU’da da çalışır.

---

## 🎯 Amaç

Bu repo “production level” bir ürün değil;  
CNN ve görüntü sınıflandırma temellerini öğrenirken:

- veri yükleme,
- normalizasyon,
- data augmentation,
- model eğitimi,
- metrik analizi,
- tekil tahmin ve model kaydetme

adımlarını uçtan uca denediğim küçük bir çalışma alanı.
