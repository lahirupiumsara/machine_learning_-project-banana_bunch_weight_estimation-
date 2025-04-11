import cv2
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import img_to_array

# Paths
DATASET_PATH = "dataset/"
CSV_PATH = "csv_data/banana_weights.csv"

def load_data(image_size=(640, 460)):
    # Read CSV file
    df = pd.read_csv(CSV_PATH)

    images = []
    weights = []

    for index, row in df.iterrows():
        img_path = os.path.join(DATASET_PATH, row["Image Name"])

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            img = img_to_array(img) / 255.0  # Normalize
            images.append(img)
            weights.append(row["Weight"])

    return np.array(images), np.array(weights)

if __name__ == "__main__":
    X, y = load_data()
    print(f"Loaded {len(X)} images and {len(y)} weight values.")
