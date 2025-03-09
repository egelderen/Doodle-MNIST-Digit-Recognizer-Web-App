# Doodle: MNIST Digit Recognizer Web App

This repository contains a Flask-based web application that trains a Convolutional Neural Network (CNN) on the MNIST dataset for handwritten digit recognition. It provides both a simple web interface and an API for making predictions.

---

## Overview

The application performs the following functions:
- **Model Training:**  
  Automatically trains a CNN on the MNIST dataset if a pre-trained model is not found. The trained model is saved as `mnist_model.h5` for later use.
- **API Endpoint:**  
  Exposes a `/predict` endpoint that accepts a base64-encoded image, preprocesses it, and returns the predicted digit along with confidence levels and probabilities for all classes.
- **Web Interface:**  
  Serves an HTML page via the `/` route, allowing users to interact with the model through a browser.

---

## File Structure

- **doodle.py:**  
  Contains the main application code:
  - Imports and disables GPU usage.
  - Defines and trains a CNN using TensorFlow/Keras on the MNIST dataset.
  - Implements Flask routes for the web interface (`/`) and the prediction endpoint (`/predict`).
  - Includes logic to create a default HTML template (`templates/index.html`) if it does not exist.

- **wsgi.py:**  
  A simple WSGI entry point that imports the Flask app from `doodle.py` and runs it. This file is useful for production deployment.

---

## Requirements

Ensure you have Python 3.x installed along with the following dependencies:
- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)
- [NumPy](https://numpy.org/)
- [Pillow](https://pillow.readthedocs.io/)

Install the required packages using pip:

```bash
pip install tensorflow flask numpy pillow
