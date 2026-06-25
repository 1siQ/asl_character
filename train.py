"""
ASL Alphabet — YOLO11m-cls Sınıflandırma Eğitimi
Dataset: Kaggle grassknoted/asl-alphabet (87k görüntü, 29 sınıf)

Neden YOLO11m-cls?
  - YOLO11n-cls (Nano) ile ilk denemede doğruluk %60-70 civarında kaldı.
  - Medium varyanta geçildiğinde validation top-1 accuracy %100'e ulaştı.
  - RTX 4060 Laptop GPU bu modeli 14ms/frame ile çalıştırabiliyor;
    gerçek zamanlı kullanım için yeterli.

Neden transfer learning / fine-tuning?
  - ImageNet ağırlıklarıyla başlamak, sıfırdan eğitime kıyasla çok daha az
    veri ve süreyle yüksek doğruluk elde etmeyi sağlar.
  - ASL el işaretleri genel görsel özellikler (kenar, doku, şekil) içerdiğinden
    ImageNet özellik çıkarıcısı iyi transfer olmaktadır.

Çalıştırmadan önce:
  1. https://www.kaggle.com/datasets/grassknoted/asl-alphabet adresinden
     asl_alphabet_train.zip dosyasını indirip bu klasöre çıkarın.
  2. python train.py
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ── AYARLAR ──────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("asl_alphabet_train") / "asl_alphabet_train"
OUTPUT_DIR   = Path("runs")
EPOCHS       = 50
BATCH        = 32
IMG_SIZE     = 320
MODEL_BASE   = "yolo11m-cls.pt"

# A-Z + SPACE (→ YUMRUK olarak yeniden adlandırılacak)
# NOTHING ve DELETE sınıfları kapsam dışı
KEEP_CLASSES = set([chr(i) for i in range(65, 91)] + ["SPACE"])


# ── VERİ HAZIRLIĞI ────────────────────────────────────────────────────────────

def prepare_dataset():
    """
    Ham Kaggle verisini YOLO sınıflandırma formatına dönüştürür.
    Klasör yapısı: dataset_clean/train/<SINIF>/ ve dataset_clean/val/<SINIF>/
    SPACE sınıfı YUMRUK olarak yeniden adlandırılır.
    Ön işleme: %90 train / %10 val rastgele ayrımı.
    """
    clean_dir = Path("dataset_clean")

    if clean_dir.exists():
        print(f"'{clean_dir}' zaten var, atlanıyor.")
        return clean_dir

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"\n[HATA] '{DATASET_ROOT}' bulunamadı!\n"
            "Kaggle'dan asl_alphabet_train.zip indirip bu klasöre çıkarın:\n"
            "  https://www.kaggle.com/datasets/grassknoted/asl-alphabet\n"
        )

    print("Dataset hazırlanıyor...")
    train_out = clean_dir / "train"
    val_out   = clean_dir / "val"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    for cls_dir in sorted(DATASET_ROOT.iterdir()):
        cls_name = cls_dir.name.upper()

        if cls_name not in KEEP_CLASSES:
            continue

        out_name = "YUMRUK" if cls_name == "SPACE" else cls_name

        images = sorted(cls_dir.glob("*.jpg"))
        if not images:
            images = sorted(cls_dir.glob("*.png"))

        split_at = int(len(images) * 0.9)
        splits   = {"train": images[:split_at], "val": images[split_at:]}

        for split, imgs in splits.items():
            dest = (train_out if split == "train" else val_out) / out_name
            dest.mkdir(exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  {out_name:8s}  train={len(splits['train'])}  val={len(splits['val'])}")

    print(f"\nDataset hazır → '{clean_dir}'")
    return clean_dir


# ── EĞİTİM ───────────────────────────────────────────────────────────────────

def train(dataset_dir: Path):
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\nCihaz: {'GPU' if device == 0 else 'CPU'}")
    print(f"Model: {MODEL_BASE}  |  Epoch: {EPOCHS}  |  Batch: {BATCH}  |  ImgSz: {IMG_SIZE}")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data     = str(dataset_dir),
        epochs   = EPOCHS,
        batch    = BATCH,
        imgsz    = IMG_SIZE,
        device   = device,
        project  = str(OUTPUT_DIR),
        name     = "asl_cls",
        patience = 10,
        workers  = 4,
        verbose  = True,
    )

    # En iyi ağırlıkları proje köküne kopyala
    best_pt = Path(OUTPUT_DIR) / "asl_cls" / "weights" / "best.pt"
    dest    = Path("..") / "best_cls.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, dest)
        print(f"\nModel kaydedildi → {dest.resolve()}")
    else:
        # YOLO run numarası ekleyebilir (asl_cls-2, asl_cls-3 vb.)
        candidates = sorted(Path(OUTPUT_DIR).glob("asl_cls*/weights/best.pt"))
        if candidates:
            shutil.copy2(candidates[-1], dest)
            print(f"\nModel kaydedildi → {dest.resolve()}  (kaynak: {candidates[-1]})")
        else:
            print("\n[UYARI] best.pt bulunamadı. runs/ klasörüne bakın.")

    return results


# ── ANA ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset_dir = prepare_dataset()
    train(dataset_dir)
