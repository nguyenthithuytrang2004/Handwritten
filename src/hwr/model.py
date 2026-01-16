"""
CNN Model for Handwritten Character Recognition
Supports 36 classes: 0-9, A-Z
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CLASSES)


def build_model(num_classes=NUM_CLASSES):
    """
    Build CNN model for character recognition

    Args:
        num_classes: Number of output classes

    Returns:
        tf.keras.Model: Compiled CNN model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(128, kernel_size=4, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def load_model(model_path=None):
    """
    Load trained model from file

    Args:
        model_path: Path to model file, if None try common locations

    Returns:
        tf.keras.Model: Loaded model
    """
    candidates = [
        model_path,
        Path("models") / "model_v2_emnist_finetuned.h5",  # prefer fine-tuned if present
        Path("models") / "model_v2.h5",
        Path("models") / "model.h5",
        Path("model_v2.h5"),
        Path("model.h5"),
    ]

    if model_path is None:
        candidates = candidates[1:]  # Skip None

    for path in candidates:
        if path and path.exists():
            try:
                model = tf.keras.models.load_model(path)
                print(f"✅ Loaded model from: {path}")
                return model
            except ValueError as exc:
                if "Kernel shape must have the same length" not in str(exc):
                    raise
                print(f"⚠️  Loading weights from: {path}")
                model = build_model(len(CLASSES))
                model.load_weights(path)
                return model

    raise FileNotFoundError("No model file found. Expected model_v2.h5 or model.h5 in models/ or root.")


def decode_prediction(prediction):
    """
    Decode model prediction to character

    Args:
        prediction: Model prediction array or index

    Returns:
        str: Predicted character
    """
    if isinstance(prediction, (list, np.ndarray)):
        if len(prediction.shape) > 1:
            prediction = np.argmax(prediction[0])
        else:
            prediction = np.argmax(prediction)

    if 0 <= prediction < len(CLASSES):
        return CLASSES[prediction]
    return "?"
