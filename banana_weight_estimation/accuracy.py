import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load model
model = load_model("models/banana_weight_model.h5", custom_objects={'mse': MeanSquaredError()})

# Load CSV
csv_path = "csv_data/banana_weights.csv"
df = pd.read_csv(csv_path)

# Folder path
image_dir = "dataset"
IMG_SIZE = (640, 460)

# Prepare data
images = []
true_weights = []

for idx, row in df.iterrows():
    image_path = os.path.join(image_dir, row['Image Name'])  # 👈 Fixed here
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype('float32') / 255.0
        images.append(img)
        true_weights.append(row['Weight'])  # This stays the same
    else:
        print(f"Image not found: {image_path}")

# Predict and evaluate
images = np.array(images)
true_weights = np.array(true_weights)

predictions = model.predict(images).flatten()

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((true_weights - predictions) / true_weights)) * 100

# Calculate accuracy (inverse of MAPE)
accuracy = 100 - mape

# Print total accuracy
print(f"\n✅ Total Accuracy: {accuracy:.2f}%")
