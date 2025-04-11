import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from preprocess import load_data  # Import function to load dataset

# Load dataset
X, y = load_data()

# Define CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(640, 460, 3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

# Train model
model = create_model()
model.summary()

history = model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2)

# Save model
model.save("models/banana_weight_model.h5")
print("Model saved successfully!")
