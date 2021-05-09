# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:11:56 2021

@author: Dhruv

Project: Animal image prediction using Transfer Learning.

Why transfer learning?
1. When you donot have resources and time to create a DL model.
2. We can use pre-trained models.

Tranfer Learning model used: VGG-16

"""

# Import libraries
from keras.applications.vgg16 import VGG16

#Instantiate model
model=VGG16()
# Display model architecture
model.summary()

# Pre-process image before feeding to model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions # shows probability of output

# Step-1: Load Image
image = load_img(r'resources/training pics/cats/cat.1.jpg')





