"""
Görüntü Sınıflandırıcı - Eğitim Scripti
=========================================
Klasörlere ayrılmış resimlerden MobileNetV2 transfer learning ile model eğitir.

Kullanım:
    python train.py                           # Varsayılan: dataset/ klasörü
    python train.py --data_dir C:\resimler    # Özel klasör yolu
    python train.py --epochs 20               # Epoch sayısını ayarla
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ---------- Sabitler ----------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
MODEL_PATH = "model.keras"
CLASS_NAMES_PATH = "class_names.json"


def build_model(num_classes: int) -> Model:
    """MobileNetV2 tabanlı sınıflandırma modeli oluşturur."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )

    # İlk aşamada base model dondurulur (sadece yeni katmanlar eğitilir)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def create_data_generators(data_dir: str):
    """Eğitim ve doğrulama veri jeneratörlerini oluşturur."""
    # Eğitim verisi: data augmentation ile
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,  # %20 doğrulama seti
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, val_generator


def train(data_dir: str, epochs: int):
    """Ana eğitim fonksiyonu."""
    print("=" * 60)
    print("  GORUNTU SINIFLANDIRICI - EGITIM")
    print("=" * 60)

    # Veri klasörü kontrol
    if not os.path.isdir(data_dir):
        print(f"\n[HATA] '{data_dir}' klasoru bulunamadi!")
        print("Lutfen resimlerinizin bulundugu klasoru --data_dir ile belirtin.")
        sys.exit(1)

    # Alt klasörleri kontrol et
    subdirs = [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    if len(subdirs) < 2:
        print(f"\n[HATA] En az 2 sinif klasoru gerekli, {len(subdirs)} bulundu!")
        print("Her sinif icin ayri bir alt klasor olusturun.")
        sys.exit(1)

    print(f"\n[>] Veri klasoru : {os.path.abspath(data_dir)}")
    print(f"[>] Sinif sayisi : {len(subdirs)}")
    print(f"[>] Siniflar     : {', '.join(sorted(subdirs))}")

    # Veri jeneratörleri
    print("\n[*] Veri yukleniyor...")
    train_gen, val_gen = create_data_generators(data_dir)

    num_classes = train_gen.num_classes
    num_train = train_gen.samples
    num_val = val_gen.samples

    print(f"    Egitim resim sayisi   : {num_train}")
    print(f"    Dogrulama resim sayisi: {num_val}")

    # Model oluştur
    print("\n[*] Model olusturuluyor (MobileNetV2)...")
    model, base_model = build_model(num_classes)

    # ========== AŞAMA 1: Sadece yeni katmanları eğit ==========
    print("\n" + "-" * 60)
    print("  ASAMA 1: Yeni katmanlarin egitimi")
    print("-" * 60)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase1 = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    phase1_epochs = min(epochs // 2, 10)
    start_time = time.time()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase1_epochs,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    phase1_time = time.time() - start_time
    print(f"\n[*] Asama 1 suresi: {phase1_time:.1f}s")

    # ========== AŞAMA 2: Fine-tuning (son katmanları aç) ==========
    print("\n" + "-" * 60)
    print("  ASAMA 2: Fine-tuning")
    print("-" * 60)

    # Son 30 katmanı eğitime aç
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    remaining_epochs = epochs - phase1_epochs
    start_time = time.time()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=remaining_epochs,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    phase2_time = time.time() - start_time
    print(f"\n[*] Asama 2 suresi: {phase2_time:.1f}s")

    # ========== SONUÇLAR ==========
    print("\n" + "=" * 60)
    print("  EGITIM TAMAMLANDI")
    print("=" * 60)

    # Doğrulama seti üzerinde değerlendirme
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n[+] Dogrulama dogrulugu : %{val_acc * 100:.1f}")
    print(f"[+] Dogrulama kaybi    : {val_loss:.4f}")

    # Sınıf isimleri
    class_names = {v: k for k, v in train_gen.class_indices.items()}

    # Modeli kaydet
    model.save(MODEL_PATH)
    print(f"\n[+] Model kaydedildi     : {os.path.abspath(MODEL_PATH)}")

    # Sınıf isimlerini kaydet
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"[+] Sinif isimleri       : {os.path.abspath(CLASS_NAMES_PATH)}")

    total_time = phase1_time + phase2_time
    print(f"\n[*] Toplam egitim suresi: {total_time:.1f}s")
    print(f"\n[OK] Artik 'python predict.py resim.jpg' ile tahmin yapabilirsiniz!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Klasor yapisindaki resimlerden model egitir."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Resimlerin bulundugu ana klasor (varsayilan: dataset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Toplam epoch sayisi (varsayilan: {DEFAULT_EPOCHS})",
    )
    args = parser.parse_args()

    train(args.data_dir, args.epochs)
