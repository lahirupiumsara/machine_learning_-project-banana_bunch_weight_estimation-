import tensorflow as tf
import cv2
import numpy as np

# Custom Mean Squared Error metric if it was used during training
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the pre-trained model with the custom metric
model = tf.keras.models.load_model("models/banana_weight_model.h5", custom_objects={"mse": mse})

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError("Image not found or unable to load image.")
    
    # Resize the image to match the model input
    img = cv2.resize(img, (640, 460))  # Adjust if your model requires a different size
    img = img.astype('float32') / 255.0  # Normalize image
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Function to predict the weight
def predict_weight(image_path):
    try:
        # Preprocess the image
        img = preprocess_image(image_path)
        
        # Make prediction
        predicted_weight = model.predict(img)
        
        # Assuming the model outputs weight in grams directly
        return predicted_weight[0][0]  # Extract the predicted weight value

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Example usage
if __name__ == "__main__":
    test_image_path = 'static/uploads/test_banana.jpg'  # Path to your test image
    predicted_weight = predict_weight(test_image_path)
    
    if predicted_weight is not None:
        print(f"Predicted Weight: {predicted_weight} grams")
    else:
        print("Failed to predict weight.")
