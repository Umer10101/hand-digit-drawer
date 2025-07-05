import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to 0-1 range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Check if trained model already exists
model_path = 'digit_recognition_model.h5'
import os

if os.path.exists(model_path):
    print("Loading existing trained model...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
else:
    print("No existing model found. Training new model...")

    # Build CNN model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Display model architecture
    model.summary()

    # Train the model
    print("Training the model...")
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        verbose=1)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the trained model
    model.save(model_path)
    print(f"Model saved as '{model_path}'")


# Function to preprocess hand-drawn image
def preprocess_image(image_path):
    """
    Preprocess a hand-drawn digit image to match MNIST format
    """
    # Load image
    img = Image.open(image_path)

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Invert colors if needed (MNIST has white digits on black background)
    # Check if background is mostly white (hand-drawn on white paper)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize to 0-1 range
    img_array = img_array.astype('float32') / 255.0

    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


# Function to predict digit
def predict_digit(image_path):
    """
    Predict the digit in a hand-drawn image
    """
    # Preprocess image
    processed_img = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display results
    plt.figure(figsize=(8, 4))

    # Show processed image
    plt.subplot(1, 2, 1)
    plt.imshow(processed_img.reshape(28, 28), cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')

    # Show prediction probabilities
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction[0])
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Prediction: {predicted_digit} (Confidence: {confidence:.2f})')
    plt.xticks(range(10))

    plt.tight_layout()
    plt.show()

    return predicted_digit, confidence


# Predict digit from your hand-drawn image
image_path = 'hand_image_2.jpg'

try:
    predicted_digit, confidence = predict_digit(image_path)
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}")
except FileNotFoundError:
    print(f"Image file '{image_path}' not found. Please make sure the file exists.")
except Exception as e:
    print(f"Error processing image: {e}")


# Alternative: Load pre-trained model if you already have one saved
def load_and_predict(model_path, image_path):
    """
    Load a pre-trained model and predict digit
    """
    try:
        loaded_model = keras.models.load_model(model_path)
        processed_img = preprocess_image(image_path)
        prediction = loaded_model.predict(processed_img)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_digit, confidence
    except Exception as e:
        print(f"Error loading model or predicting: {e}")
        return None, None

# Example usage with saved model:
# predicted_digit, confidence = load_and_predict('digit_recognition_model.h5', 'hand_image.jpg')