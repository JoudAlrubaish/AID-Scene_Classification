
"""
This file contains shared utilities for the AID Scene Classification project.

It contains the common settings, data loading pipeline, random seed setup, and CNN builder used by all models.

Using shared functions ensures that Models A, B, C, and D use the same preprocessing and experimental settings for fair comparison.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import tensorflow as tf


# Shared project settings
SEED = 42
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15

PROJECT_ROOT = Path(__file__).resolve().parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)


TARGET_CLASSES = [
    "Forest",
    "Desert",
    "Beach",
    "River",
    "Mountain",
    "DenseResidential",
    "Industrial",
    "Airport",
    "Stadium",
    "Farmland",
]

CLASS_TO_IDX = {
    class_name: index
    for index, class_name in enumerate(TARGET_CLASSES)
}


def set_seed():
    """Set random seeds to make experiments reproducible."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def make_dataset(csv_path, shuffle=False):
    """
    Create a TensorFlow dataset from a split CSV file.

    Images are loaded, resized to the common image size,
    mapped to numerical labels, batched, and prefetched.
    """

    df = pd.read_csv(csv_path)

    filepaths = [
        str(PROJECT_ROOT / path)
        for path in df["filepath"]
    ]

    labels = [
        CLASS_TO_IDX[label]
        for label in df["label"]
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels)
    )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)

        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )

        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(filepaths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_datasets():
    """
    Load the fixed training, validation, and test datasets.

    The same saved data splits are used by all CNN models
    to ensure a fair comparison.
    """
    train_ds = make_dataset(
        SPLITS_DIR / "train.csv",
        shuffle=True,
    )

    val_ds = make_dataset(
        SPLITS_DIR / "val.csv",
        shuffle=False,
    )

    test_ds = make_dataset(
        SPLITS_DIR / "test.csv",
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


def build_cnn(num_classes, config):
    """
    The configuration controls the convolutional filters,
    kernel size, augmentation, batch normalization,
    dense layer size, and dropout rate.

    This builder is shared across the project so that model
    architectures can be changed while keeping the same structure.
    """
    inputs = tf.keras.Input(
        shape=(*IMG_SIZE, 3)
    )

    x = tf.keras.layers.Rescaling(
        1.0 / 255
    )(inputs)

    if config.get("augmentation", False):
        x = tf.keras.layers.RandomFlip(
            "horizontal"
        )(x)

    batch_norm = config.get(
        "batch_norm",
        False,
    )

    for filters in config["filters"]:
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=config["kernel_size"],
            padding="same",
            use_bias=not batch_norm,
        )(x)

        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    dense_units = config.get(
        "dense_units",
        0,
    )

    if dense_units > 0:
        x = tf.keras.layers.Dense(
            dense_units,
            activation="relu",
        )(x)

    dropout = config.get(
        "dropout",
        0.0,
    )

    if dropout > 0:
        x = tf.keras.layers.Dropout(
            dropout
        )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name=config["name"],
    )