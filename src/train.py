# --- Standard Library ---
import argparse          # Parse command-line options like --data_dir, --epochs, etc.
from pathlib import Path # Safer path handling than raw strings (works on all OSes)
import os                # Interact with the filesystem (env vars, directories)
import sys               # Access interpreter details (version, exit codes), optional

# --- Third-Party: Numerical / ML ---
import numpy as np       # Efficient arrays, math ops, and utilities used by ML code
import tensorflow as tf  # Deep-learning framework powering our model/training

# --- Keras High-Level APIs (bundled with TensorFlow) ---
from tensorflow.keras import layers, models
#  - layers: building blocks (Conv2D, Dense, Dropout, etc.)
#  - models: to assemble layers into a Model object (Model/Sequential)

from tensorflow.keras.applications import MobileNetV2
#  - Predefined CNN architecture we’ll fine-tune for currency recognition

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#  - Input normalization function exactly matching MobileNetV2’s training regime
#    (important for getting good accuracy)

from tensorflow.keras.preprocessing import image_dataset_from_directory
#  - Utility to create batched tf.data datasets directly from a folder structure

from tensorflow.keras.callbacks import (
    ModelCheckpoint,   # Save the best model (e.g., highest val_accuracy) during training
    EarlyStopping,     # Stop training when improvement stalls (prevents overfitting)
    ReduceLROnPlateau  # Lower learning rate when progress slows (helps converge)
)
