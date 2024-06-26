import streamlit as sl
import tensorflow
from tensorflow import keras
import numpy as np
import requests
from PIL import Image
from io import BytesIO

#def disp_img(url):
    #response = requests.get(url)
    #img = Image.open(BytesIO(response.content))
    #resized = img.resize[512,512]
    #img_arr = np.array(resized)
    #scaled_img = img_arr/255
    #sl.image(scaled_img,caption='Input Image')
    #return scaled_img

try:
    cnn_model = keras.models.load_model('cnn_model.h5')
except Exception as ex:
    sl.error('Cannot able to load this model because of:',ex)
    sl.stop()

labels_classes = ['Airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def load_and_show(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    resized = img.resize([32,32])
    img_arr = np.array(resized)
    scaled_img = img_arr/255
    resized1 = img.resize([512,512])
    img_arr1 = np.array(resized1)
    sl.image(img_arr1,caption='Input image',use_column_width=True)
    return scaled_img

def predict(scaled_img):
    batch_count = np.expand_dims(scaled_img,axis=0)
    preds_prob = cnn_model.predict(batch_count)
    category = np.argmax(preds_prob)
    sl.write('The image is of:',labels_classes[category])

url = sl.text_input('Enter the url from browser to classify an image')

if sl.button('Make prediction'):
        try:
            scaled_img = load_and_show(url)
            predict(scaled_img)
        except requests.exceptions.MissingSchema:
            sl.error('Invalud url. Please include http:// or https://')
        except Exception as e:
            sl.error('An exception is occurred:',{e})
