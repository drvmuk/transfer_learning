# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:11:56 2021

@author: Dhruv

Project: Animal image prediction using Transfer Learning.

Why transfer learning?
1. When you donot have resources and time to create a DL model.
2. We can use pre-trained models.

Transfer Learning model used: VGG-16
VGG-16 works on 1000 classes

"""

# Import libraries
import numpy as np
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

data_path = r'resources\training pics'
data = os.listdir(data_path)

img_data_list = []

for dataset in data:
    img_list = os.listdir(data_path + '/' + dataset)
    print('loaded images' + f'{dataset}\n')
    for img in img_list:
        img_path = data_path + '/' + dataset + '/' + img
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        dim_expansion = np.expand_dims(img_array, axis=0)
        pre_input = preprocess_input(dim_expansion)
        print('Input image shape', pre_input.shape)
        img_data_list.append(pre_input)

img_data = np.array(img_data_list)
print(img_data.shape)

img_data = np.rollaxis(img_data, 1, 0)
print(img_data.shape)

img_data = img_data[0]
print(img_data.shape)

num_classes = 3
num_of_samples = img_data.shape[0]
labels = np.ones((606,), dtype='int64')

labels[0:202] = 0
labels[202:404] = 1
labels[404:606] = 2

names=['cats','dogs','horses']

Y = np_utils.to_categorical(labels, num_classes)

x,y = shuffle(img_data, Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

image_input = Input(shape = (224,224,3))
model = VGG16(input_tensor = image_input, weights = 'imagenet')
model.summary()
last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-1]:
    layer.trainable = False

print(custom_vgg_model.layers[3].trainable)

custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

t = time.time()
hist = custom_vgg_model.fit(X_train, y_train, batch_size = 32, epochs=1, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' %(t-time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size = 10, verbose = 1)

print('loss={:.4f}, accuracy:{:.4f}%'.format(loss, accuracy * 100))








