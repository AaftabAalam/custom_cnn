# CIFAR-10 Image Classification Project

This project classifies images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) model. The application is built with Streamlit and allows users to input an image URL, which is then processed and classified into one of the 10 CIFAR-10 classes.

## Libraries Used

- **Streamlit**: For creating the web application interface.
- **TensorFlow**: For loading the pre-trained CNN model.
- **NumPy**: For numerical operations.
- **Requests**: For fetching images from URLs.
- **Pillow**: For image processing.

## Project Features

- **URL Input**: Users can input the URL of an image they want to classify.
- **Image Processing**: The image is fetched from the URL, resized, and preprocessed for the model.
- **Prediction**: The pre-trained CNN model predicts the class of the image.
- **User Interface**: The application has a button to process the image and display the prediction.

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/AaftabAalam/custom_cnn.git
    cd cistom_cnn
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

4. Open your web browser and go to `http://localhost:8501` to view the application.

## Custom Model

The custom CNN model is used to classify images into one of the following CIFAR-10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Code Explanation

Here is a brief overview of the main parts of the code:

### Loading the Model

```python
import streamlit as sl
from tensorflow import keras
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Load the model
try:
    cnn_model = keras.models.load_model('cnn_model.h5')
except Exception as ex:
    sl.error(f'Cannot load the model because of: {ex}')
    sl.stop()
