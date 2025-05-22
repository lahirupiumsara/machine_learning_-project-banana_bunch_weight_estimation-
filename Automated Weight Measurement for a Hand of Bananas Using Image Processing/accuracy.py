import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load the trained model
model = load_model("models/banana_weight_model.h5", custom_objects={'mse': MeanSquaredError()})

# Load CSV file
csv_path = "csv_data/banana_weights.csv"
df = pd.read_csv(csv_path)

# Define image directory and size
image_dir = "dataset"
IMG_SIZE = (640, 460)

# Prepare image and weight data
images = []
true_weights = []

for idx, row in df.iterrows():
    image_path = os.path.join(image_dir, row['Image Name'])
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype('float32') / 255.0
        images.append(img)
        true_weights.append(row['Weight'])
    else:
        print(f"❌ Image not found: {image_path}")

# Convert to numpy arrays
images = np.array(images)
true_weights = np.array(true_weights)

# Predict
predictions = model.predict(images).flatten()

# Evaluate
mape = np.mean(np.abs((true_weights - predictions) / true_weights)) * 100
accuracy = 100 - mape

# ✅ Print total accuracy
print(f"\n✅ Total Accuracy: {accuracy:.2f}%")
print(f"📉 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 📊 Plot true vs predicted weights
plt.figure(figsize=(10, 6))
plt.plot(true_weights, label='True Weights', color='green', marker='o')
plt.plot(predictions, label='Predicted Weights', linestyle='dashed', color='orange', marker='x')
plt.title(f'Weight Prediction - Accuracy: {accuracy:.2f}%', fontsize=14)
plt.xlabel('Sample Index')
plt.ylabel('Weight (grams)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
