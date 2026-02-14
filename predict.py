"""
Görüntü Sınıflandırıcı - Tahmin Scripti
=========================================
Eğitilmiş model ile yeni resimlerin türünü tahmin eder.

Kullanım:
    python predict.py resim.jpg
    python predict.py C:\\resimler\\test.png
    python predict.py resim1.jpg resim2.png resim3.jpg   # Birden fazla resim
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

# TensorFlow loglarını kapat (hız için)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")


# ---------- Sabitler ----------
IMG_SIZE = (224, 224)
MODEL_PATH = "model.keras"
CLASS_NAMES_PATH = "class_names.json"
TOP_K = 3  # En yüksek K tahmin gösterilir


def load_model_and_classes():
    """Modeli ve sınıf isimlerini yükler."""
    if not os.path.exists(MODEL_PATH):
        print(f"[HATA] Model dosyasi bulunamadi: {MODEL_PATH}")
        print("   Once 'python train.py' ile egitim yapin.")
        sys.exit(1)

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"[HATA] Sinif isimleri dosyasi bulunamadi: {CLASS_NAMES_PATH}")
        sys.exit(1)

    print("[*] Model yukleniyor...", end=" ", flush=True)
    load_start = time.time()

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    load_time = (time.time() - load_start) * 1000
    print(f"({load_time:.0f}ms)")

    return model, class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """Resmi model için ön işler."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_single(model, class_names: dict, image_path: str):
    """Tek bir resim için tahmin yapar."""
    if not os.path.exists(image_path):
        print(f"\n[HATA] Dosya bulunamadi: {image_path}")
        return

    # Ön işleme
    img_array = preprocess_image(image_path)

    # Tahmin
    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    inference_time = (time.time() - start_time) * 1000

    # Sonuçları sırala
    pred_indices = np.argsort(predictions[0])[::-1]

    # En iyi tahmin
    best_idx = pred_indices[0]
    best_class = class_names[str(best_idx)]
    best_confidence = predictions[0][best_idx] * 100

    # Dosya adı
    filename = os.path.basename(image_path)

    print(f"\n{'-' * 50}")
    print(f"[>] Dosya  : {filename}")
    print(f"[+] Tahmin : {best_class}")
    print(f"[+] Guven  : %{best_confidence:.1f}")
    print(f"[*] Sure   : {inference_time:.0f}ms")

    # Top-K sonuçları göster
    if len(class_names) > 2:
        print(f"\n   Top-{min(TOP_K, len(class_names))} tahminler:")
        for i, idx in enumerate(pred_indices[:TOP_K]):
            cls = class_names[str(idx)]
            conf = predictions[0][idx] * 100
            bar = "#" * int(conf / 5) + "." * (20 - int(conf / 5))
            print(f"   {i+1}. {cls:20s} {bar} %{conf:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Egitilmis model ile resim turu tahmini yapar."
    )
    parser.add_argument(
        "images",
        type=str,
        nargs="+",
        help="Tahmin yapilacak resim dosya yol(lar)i",
    )
    args = parser.parse_args()

    # Model yükle (bir kere)
    model, class_names = load_model_and_classes()

    # Warmup: ilk tahmin yavaş olabilir, önceden ısıtma yap
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)

    # Her resim için tahmin yap
    for image_path in args.images:
        predict_single(model, class_names, image_path)

    print(f"\n{'-' * 50}")
    print(f"[OK] Toplam {len(args.images)} resim islendi.\n")


if __name__ == "__main__":
    main()
