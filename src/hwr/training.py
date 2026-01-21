"""
Training utilities for HWR model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image

from .model import build_model, CLASSES


def load_user_data(base_path="data/user_data"):
    """
    Load user-generated training data

    Args:
        base_path: Path to user data directory

    Returns:
        tuple: (images, labels) or (None, None)
    """
    base = Path(base_path)
    if not base.exists():
        return None, None

    images = []
    labels = []

    for idx, label in enumerate(CLASSES):
        folder = base / label
        if not folder.exists():
            continue

        for img_path in folder.glob("*.png"):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L").resize((28, 28))
                    images.append(np.array(img))
                    labels.append(idx)
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
                continue

    if not images:
        return None, None

    x_user = np.stack(images, axis=0)
    y_user = np.array(labels)
    return x_user, y_user


def append_user_data(x_train, y_train, x_test, y_test, num_classes, user_data_path="data/user_data"):
    """
    Append user data to existing dataset

    Args:
        x_train, y_train, x_test, y_test: Existing train/test data
        num_classes: Number of classes
        user_data_path: Path to user data

    Returns:
        tuple: Updated (x_train, y_train, x_test, y_test, user_count)
    """
    user_data = load_user_data(user_data_path)
    if user_data is None:
        return x_train, y_train, x_test, y_test, 0

    x_user, y_user = user_data
    x_user = x_user.reshape(x_user.shape[0], 28, 28, 1)

    if x_user.shape[0] >= 20:
        _, counts = np.unique(y_user, return_counts=True)
        stratify_labels = y_user if np.all(counts >= 2) else None

        x_utrain, x_utest, y_utrain, y_utest = train_test_split(
            x_user, y_user, train_size=0.9, stratify=stratify_labels, random_state=42
        )

        y_utrain = to_categorical(y_utrain, num_classes=num_classes)
        y_utest = to_categorical(y_utest, num_classes=num_classes)

        x_train = np.concatenate((x_train, x_utrain), axis=0)
        y_train = np.concatenate((y_train, y_utrain), axis=0)
        x_test = np.concatenate((x_test, x_utest), axis=0)
        y_test = np.concatenate((y_test, y_utest), axis=0)
    else:
        y_user = to_categorical(y_user, num_classes=num_classes)
        x_train = np.concatenate((x_train, x_user), axis=0)
        y_train = np.concatenate((y_train, y_user), axis=0)

    return x_train, y_train, x_test, y_test, x_user.shape[0]


def load_mnist_data():
    """
    Load MNIST dataset from Keras

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        print(f"❌ Error loading MNIST: {e}")
        raise


def load_nist_data(data_path="data/A_Z Handwritten Data.csv"):
    """
    Load NIST handwritten letters dataset

    Args:
        data_path: Path to A_Z Handwritten Data.csv

    Returns:
        tuple: (x_data, y_data)
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"NIST data not found at {path}")

    letters_dataset = pd.read_csv(path, header=None)

    # Clean dataset
    first_row = pd.to_numeric(letters_dataset.iloc[0], errors="coerce")
    if first_row.isna().any() or np.array_equal(first_row.to_numpy(), np.arange(len(first_row))):
        letters_dataset = letters_dataset.iloc[1:].reset_index(drop=True)

    letters_dataset = letters_dataset[pd.to_numeric(letters_dataset[0], errors="coerce").notna()]
    letters_dataset[0] = letters_dataset[0].astype(int)

    # Adjust labels (NIST uses 1-26 for A-Z, we want 0-25)
    if letters_dataset[0].min() == 1 and letters_dataset[0].max() == 26:
        letters_dataset[0] = letters_dataset[0] - 1

    # Sample data to balance
    letters_dataset = letters_dataset.groupby(0).head(1000)

    y_data = letters_dataset[0].to_numpy()
    x_data = letters_dataset.drop(0, axis=1).to_numpy()

    return x_data, y_data


def prepare_datasets(x_digits, y_digits, x_letters, y_letters, num_classes):
    """
    Prepare and combine datasets

    Args:
        x_digits, y_digits: MNIST digit data
        x_letters, y_letters: NIST letter data
        num_classes: Total number of classes

    Returns:
        tuple: (x_train, x_test, y_train, y_test, num_classes)
    """
    # Convert Y to one-hot
    y_digits = to_categorical(y_digits, num_classes=num_classes)
    y_letters = to_categorical(y_letters + 10, num_classes=num_classes)  # Letters start at class 10

    # Split data
    x_train1, x_test1, y_train1, y_test1 = train_test_split(
        x_digits, y_digits, train_size=0.9, stratify=y_digits.argmax(axis=1), random_state=42
    )
    x_train2, x_test2, y_train2, y_test2 = train_test_split(
        x_letters, y_letters, train_size=0.9, stratify=y_letters.argmax(axis=1), random_state=42
    )

    # Reshape to CNN input
    x_train1 = x_train1.reshape(x_train1.shape[0], 28, 28, 1)
    x_test1 = x_test1.reshape(x_test1.shape[0], 28, 28, 1)
    x_train2 = x_train2.reshape(x_train2.shape[0], 28, 28, 1)
    x_test2 = x_test2.reshape(x_test2.shape[0], 28, 28, 1)

    # Combine datasets
    x_train = np.concatenate((x_train1, x_train2), axis=0)
    x_test = np.concatenate((x_test1, x_test2), axis=0)
    y_train = np.concatenate((y_train1, y_train2), axis=0)
    y_test = np.concatenate((y_test1, y_test2), axis=0)

    return x_train, x_test, y_train, y_test, num_classes


def load_full_dataset():
    """
    Load and prepare full training dataset

    Returns:
        tuple: (x_train, x_test, y_train, y_test, num_classes)
    """
    print("🔄 Loading MNIST digits...")
    x_digits, x_digits_test, y_digits, y_digits_test = load_mnist_data()

    # Combine train/test for more data
    x_digits = np.concatenate((x_digits, x_digits_test), axis=0)
    y_digits = np.concatenate((y_digits, y_digits_test), axis=0)

    # Sample 1000 per class
    indices = []
    for label in range(10):
        label_indices = np.where(y_digits == label)[0]
        if len(label_indices) < 1000:
            raise ValueError(f"Not enough MNIST samples for digit {label}")
        indices.append(np.random.choice(label_indices, size=1000, replace=False))
    indices = np.concatenate(indices)

    x_digits = x_digits[indices]
    y_digits = y_digits[indices]

    print("🔄 Loading NIST letters...")
    x_letters, y_letters = load_nist_data()

    num_classes = 36  # 10 digits + 26 letters

    print("🔄 Preparing datasets...")
    return prepare_datasets(x_digits, y_digits, x_letters, y_letters, num_classes)


def train_model(model=None, epochs=30, batch_size=64, user_data_path="data/user_data"):
    """
    Train the HWR model

    Args:
        model: Existing model to fine-tune, if None build new
        epochs: Number of training epochs
        batch_size: Batch size
        user_data_path: Path to user data for fine-tuning

    Returns:
        tf.keras.Model: Trained model
    """
    # Load data
    x_train, x_test, y_train, y_test, num_classes = load_full_dataset()

    # Append user data if available
    x_train, y_train, x_test, y_test, user_count = append_user_data(
        x_train, y_train, x_test, y_test, num_classes, user_data_path
    )

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build or load model
    if model is None:
        model = build_model(num_classes)

    # Compile
    learning_rate = 1e-4 if user_count > 0 else 1e-3  # Lower LR for fine-tuning
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    # Data augmentation
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    # Train
    steps_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))

    print(f"🚀 Training model with {x_train.shape[0]} samples...")
    if user_count > 0:
        print(f"📚 Including {user_count} user samples for fine-tuning")

    history = model.fit(
        generator.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test, y_test),
        verbose=2
    )

    return model, history
