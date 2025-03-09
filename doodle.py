import os
import numpy as np
# Disable GPU before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import flask
from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io

# Create Flask app
app = Flask(__name__)

# Global variable for the model
model = None

# Part 1: Train the MNIST model
def train_mnist_model(save_path='mnist_model.h5'):
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape data for CNN input
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    
    # One-hot encode targets
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Build the model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))
    
    # Evaluate the model
    score = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {score[1]}")
    
    # Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model

# Ensure the model is loaded before the first request
def load_trained_model():
    global model
    model_path = 'mnist_model.h5'
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_model(model_path)
    else:
        print("Training new MNIST model...")
        model = train_mnist_model(model_path)
    
    print("Model loaded and ready!")

# Call load_trained_model when the module is imported
load_trained_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure model is loaded
    global model
    if model is None:
        load_trained_model()
    
    # Get image data from POST request
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    
    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Resize and preprocess image
    image = image.resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255
    
    # Make prediction using global model
    prediction = model.predict(image_array)
    result = np.argmax(prediction[0])
    confidence = float(prediction[0][result]) * 100
    
    return jsonify({
        'digit': int(result),
        'confidence': confidence,
        'probabilities': prediction[0].tolist()
    })

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Write HTML template to file if it doesn't exist
template_path = 'templates/index.html'
if not os.path.exists(template_path):
    with open(template_path, 'w') as f:
        # The existing HTML content from the original script remains the same
        f.write("""INSERT_THE_ENTIRE_HTML_CONTENT_HERE""")  # You would replace this with the full HTML content from the original script