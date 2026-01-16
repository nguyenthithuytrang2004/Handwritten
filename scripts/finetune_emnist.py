#!/usr/bin/env python3
"""
Download EMNIST (letters), preprocess, combine with existing A_Z data/user samples,
and fine-tune the existing HWR model.

Usage:
  python scripts/finetune_emnist.py --epochs 5 --batch-size 64

Notes:
 - Requires `tensorflow-datasets`. Install with `pip install tensorflow-datasets`.
 - Designed to run on CPU/GPU depending on your TensorFlow installation.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Ensure src is importable (robustly add project/src to sys.path)
project_root = Path(__file__).resolve().parent.parent
# Add project root so `import src...` works (we keep package name 'src')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
else:
    sys.path[0] = str(project_root)


try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except Exception as e:
    print("❌ Missing dependency:", e)
    print("💡 Install required packages: pip install tensorflow tensorflow-datasets")
    raise
else:
    # Reduce TF/absl verbosity to avoid noisy terminal output
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # ERROR and above
    try:
        import logging
        tf.get_logger().setLevel("ERROR")
        logging.getLogger("absl").setLevel(logging.ERROR)
    except Exception:
        pass

from src.hwr.model import load_model, build_model, CLASSES
from src.hwr.training import load_nist_data, load_user_data
from tensorflow.keras.utils import to_categorical


def emnist_to_numpy(split_name="train"):
    ds = tfds.load("emnist/letters", split=split_name, as_supervised=True)
    images = []
    labels = []
    for img, label in tfds.as_numpy(ds):
        # img: (28,28,1) uint8
        arr = img.squeeze()
        # EMNIST images are transposed/rotated relative to standard MNIST — rotate to upright
        arr = np.rot90(arr, k=1)  # try rotate; this generally aligns characters
        arr = np.fliplr(arr)
        images.append(arr)
        labels.append(int(label))
    return np.stack(images, axis=0), np.array(labels)


def prepare_emnist_for_model(x, y):
    # EMNIST 'letters' labels: 1..26 -> convert to 0..25 then map to 10..35 (A-Z)
    y = y - 1  # 0..25
    y = y + 10  # map to 10..35 to align with CLASSES "0-9 A-Z"
    num_classes = len(CLASSES)
    y_cat = to_categorical(y, num_classes=num_classes)
    x = x.reshape((x.shape[0], 28, 28, 1)).astype("float32") / 255.0
    return x, y_cat


def load_a_z_nist():
    # load_nist_data returns X (pixels flattened) and y (0..25)
    x_letters, y_letters = load_nist_data()
    # x_letters shape: (N, 784)
    x_letters = x_letters.reshape((-1, 28, 28)).astype("float32")
    # NIST labels are 0..25 already; map to 10..35
    y_letters = y_letters + 10
    num_classes = len(CLASSES)
    y_cat = to_categorical(y_letters, num_classes=num_classes)
    x_letters = x_letters.reshape((x_letters.shape[0], 28, 28, 1)) / 255.0
    return x_letters, y_cat


def load_user_samples():
    user = load_user_data()
    if user is None or user[0] is None:
        return None, None
    x_u, y_u = user
    x_u = x_u.reshape((x_u.shape[0], 28, 28, 1)).astype("float32") / 255.0
    num_classes = len(CLASSES)
    y_u = to_categorical(y_u, num_classes=num_classes)
    return x_u, y_u


def build_combined_dataset(emnist_x, emnist_y, nist_x, nist_y, user_x=None, user_y=None):
    X = np.concatenate((emnist_x, nist_x), axis=0)
    Y = np.concatenate((emnist_y, nist_y), axis=0)
    if user_x is not None:
        X = np.concatenate((X, user_x), axis=0)
        Y = np.concatenate((Y, user_y), axis=0)
    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], Y[idx]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", default=None, help="Path to existing model file (optional)")
    args = parser.parse_args(argv)

    print("🔄 Downloading EMNIST (letters)...")
    x_em, y_em = emnist_to_numpy("train")
    x_em, y_em = prepare_emnist_for_model(x_em, y_em)
    print(f"  EMNIST: {x_em.shape[0]} samples")

    print("🔄 Loading NIST A_Z data...")
    x_nist, y_nist = load_a_z_nist()
    print(f"  NIST letters: {x_nist.shape[0]} samples")

    print("🔄 Loading user samples (if any)...")
    x_user, y_user = load_user_samples()
    if x_user is not None:
        print(f"  User samples: {x_user.shape[0]} samples")
    else:
        print("  No user samples found.")

    print("🔄 Combining datasets...")
    X, Y = build_combined_dataset(x_em, y_em, x_nist, y_nist, x_user, y_user)
    # Split train/test
    split = int(0.9 * len(X))
    x_train, y_train = X[:split], Y[:split]
    x_val, y_val = X[split:], Y[split:]

    print(f"📚 Training samples: {x_train.shape[0]}, Validation: {x_val.shape[0]}")

    # Load or build model
    try:
        model = load_model(args.model_path)
    except Exception:
        print("⚠️ Could not load existing model, building new model.")
        model = build_model(len(CLASSES))

    lr = 1e-4
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr), metrics=["accuracy"])

    # Augmentation: rotation, shift, zoom, shear, brightness jitter
    gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=5.0,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest"
    )
    steps = int(np.ceil(x_train.shape[0] / args.batch_size))

    print(f"🚀 Fine-tuning for {args.epochs} epochs (lr={lr})...")
    history = model.fit(gen.flow(x_train, y_train, batch_size=args.batch_size), epochs=args.epochs, steps_per_epoch=steps, validation_data=(x_val, y_val), verbose=2)

    out_path = os.path.join("models", "model_v2_emnist_finetuned.h5")
    model.save(out_path)
    print(f"✅ Fine-tuned model saved to {out_path}")
    # Also update canonical model files so GUI loads the new model automatically
    try:
        canonical_path = os.path.join("models", "model_v2.h5")
        model.save(canonical_path)
        print(f"✅ Canonical model updated at {canonical_path}")
        backup_path = os.path.join("models", "model.h5")
        model.save(backup_path)
        print(f"✅ Backup model saved at {backup_path}")
    except Exception as e:
        print(f"⚠️ Could not write canonical model copies: {e}")

    # Print final metrics
    acc = history.history.get("accuracy", [])[-1] if history.history.get("accuracy") else None
    val_acc = history.history.get("val_accuracy", [])[-1] if history.history.get("val_accuracy") else None
    print(f"Final training accuracy: {acc:.4f}" if acc is not None else "")
    print(f"Final validation accuracy: {val_acc:.4f}" if val_acc is not None else "")


if __name__ == "__main__":
    main()


