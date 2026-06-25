# ASL Character Recognition

TeknoGenç TeknoLAB 2025-2026 dönem sonu projesi. Amerikan İşaret Dili (ASL) alfabesini web
kamerasından gerçek zamanlı olarak tanıyıp metne dönüştüren, tamamen çevrimdışı çalışan bir
yapay zeka sistemidir.

## Problem Tanımı

Türkiye'de yaklaşık 3 milyon işitme engelli birey bulunmaktadır. Bu bireyler günlük iletişimde
ciddi engellerle karşılaşmaktadır. Mevcut çözümler ya pahalı uzman tercümanlara ya da internet
bağlantısı gerektiren bulut servislerine dayanmaktadır. Bu proje, internet bağlantısı olmadan
çalışan, düşük maliyetli bir alternatif sunar.

## Çözüm

- YOLO11m-cls modeli 87.000 görüntü ile fine-tune edildi (27 sınıf: A–Z + YUMRUK)
- ASLController adlı sonlu durum makinesi (FSM) anlık tespitleri kararlı kelimelere dönüştürür
- Çevrimdışı NLP ile kelime tamamlama önerisi üretilir
- İki arayüz: Streamlit web uygulaması (Hugging Face Spaces) + PyQt5 masaüstü uygulaması

## Proje Yapısı

```
proje_son/
├── app.py                    # Streamlit web uygulaması (HF Spaces / yerel)
├── train.py                  # YOLO11m-cls eğitim scripti
├── pyqt_app/
│   └── main.py               # PyQt5 masaüstü uygulaması (yüksek FPS, yerel)
├── training_results/
│   ├── results.png           # Loss ve accuracy eğrileri
│   ├── confusion_matrix.png  # Karışıklık matrisi
│   ├── confusion_matrix_normalized.png
│   ├── val_batch0_pred.jpg   # Örnek validation tahminleri
│   └── train_batch0.jpg      # Örnek eğitim batch
├── requirements.txt          # Streamlit / HF Spaces (CPU uyumlu)
└── requirements_local.txt    # Yerel geliştirme (CUDA + PyQt5)
```

## Kurulum

### Streamlit Web Uygulaması

```bash
git clone https://github.com/1siQ/asl_character.git
cd asl_character

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

pip install -r requirements.txt
```

`best_cls.pt` model dosyasını kök dizine ekleyin, ardından:

```bash
streamlit run app.py
```

### PyQt5 Masaüstü Uygulaması (Yerel, CUDA)

```bash
pip install torch==2.12.1+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_local.txt
python pyqt_app/main.py
```

### Modeli Yeniden Eğitmek

```bash
# Kaggle'dan dataset indir: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
# asl_alphabet_train/ klasörünü proje_son/ içine çıkar
python train.py
```

## Model ve Eğitim

### Mimari Seçimi: Neden YOLO11m-cls?

| Kriter | YOLO11n-cls (Denendi) | YOLO11m-cls (Seçildi) |
|---|---|---|
| Val Top-1 Acc | ~%60–70 | %100 |
| Inference Süresi | ~8 ms | ~14 ms |
| Model Boyutu | 5 MB | 21 MB |
| GPU Gereksinimi | Çok Düşük | Orta |

YOLO11n-cls (Nano) yetersiz doğruluk verdi. RTX 4060 GPU 14 ms inference süresini gerçek
zamanlı olarak kaldırabildiğinden ve doğruluk hızdan öncelikli olduğundan YOLO11m-cls seçildi.

### Yöntem: Neden Fine-tuning / Transfer Learning?

- ImageNet ağırlıklarıyla başlamak, sıfırdan eğitime kıyasla çok daha az veri ile yüksek
  doğruluk elde etmeyi sağlar.
- ASL el işaretleri genel görsel özellikler (kenar, doku, şekil) içerdiğinden ImageNet
  özellik çıkarıcısı iyi transfer olmaktadır.

### Eğitim Parametreleri

| Parametre | Değer |
|---|---|
| Model | YOLO11m-cls (pretrained ImageNet) |
| Epoch | 50 (patience=10 ile ~15-20'de durdu) |
| Batch Size | 32 |
| Görüntü Boyutu | 320×320 px |
| Optimizer | SGD (momentum=0.937, weight_decay=0.0005) |
| Donanım | NVIDIA RTX 4060 Laptop GPU, CUDA 12.6 |
| Eğitim Süresi | ~18 dakika |

### Veri Seti

- Kaynak: [Kaggle — grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Boyut: 87.000 görüntü, 29 orijinal sınıf
- Kullanılan: 27 sınıf (A–Z + SPACE → YUMRUK olarak yeniden adlandırıldı; NOTHING ve DELETE dışarıda)
- Format: JPEG, ~200×200 px, beyaz/açık arka plan
- Ön işleme: %90 train / %10 val ayrımı; YOLO dahili augmentation (renk jitter, ölçek, çevirme)

Neden bu veri seti? Açık kaynaklı ASL veri setleri arasında sınıf başına en dengeli ve en
büyük örneklem sayısına sahip (sınıf başına ~3.000 görüntü).

## Performans

| Metrik | Değer |
|---|---|
| Train Top-1 Accuracy | %100 |
| Val Top-1 Accuracy | %100 |
| Güven Eşiği (inference) | >= 0.80 |
| Gerçek Zamanlı Hız (PyQt5) | ~30 FPS |
| Gerçek Zamanlı Hız (Streamlit) | ~15 FPS |

Görsel kanıtlar için `training_results/` klasörüne bakın.

### Modelin Zayıf Noktaları

- Veri seti kontrollü stüdyo ortamında çekildi; değişken aydınlatma ve karmaşık arka
  planlar doğruluğu düşürebilir.
- M/N, R/U gibi benzer el şekillerine sahip harfler kullanıcı pozisyonuna bağlı olarak
  karıştırılabilir.

## ASLController (FSM)

Anlık YOLO tespitlerini kararlı harf/kelime çıktısına dönüştüren sonlu durum makinesi:

| Durum | Açıklama |
|---|---|
| IDLE | Aktif sinyal yok |
| LOCKING | Kayan pencerede oy toplanıyor |
| WRITTEN | Harf yazıldı, aynı harf kilitleniyor (latch) |

Commit kuralı: Son 20 karenin 14'ünde (%70) aynı sınıf görülmesi gerekir.
Latch: Yazılan harf kilitlenir; el çerçeveden çıkmadan (20 null kare) aynı harf tekrar yazılamaz.
Çift harf için: YUMRUK (yumruk gesturesi) latch'i sıfırlar.

## Arayüz Karşılaştırması

| Özellik | Streamlit (app.py) | PyQt5 (pyqt_app/main.py) |
|---|---|---|
| Erişim | Web tarayıcı, HF Spaces | Yerel masaüstü |
| FPS | ~15 | ~30 |
| Swipe Tespiti | Hayır (butonlar) | Evet (optik akış) |
| Kurulum | pip + streamlit run | pip + python |
| Demo kolaylığı | Yüksek | Orta |

## Gereksinimler

- Python 3.10+
- CUDA destekli GPU önerilir (CPU ile de çalışır, daha yavaş)
- `best_cls.pt` model dosyası (boyut nedeniyle repoya dahil edilmemiştir)

## Lisans

MIT
