#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tqdm import tqdm
import os
import cv2
import numpy as np
from random import randrange
from keras.utils import to_categorical


# In[7]:


def get_label(path):
   piece = os.path.basename(path).split('_')[0]
   d = {'0': 0, 'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,           'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12}
   return d[piece]

def get_img(path):
   return cv2.resize(cv2.imread(path), (224,224))

def get_data(path_dir):
   images = []
   labels = []
   for root, dirs, files in (os.walk(path_dir)):
       for file in files:
           if 'jpg' not in file: continue
           path = os.path.join(root, file)
           image = get_img(path)
           label = get_label(path)
           images.append(image)
           labels.append(label)
   images = np.array(images)/255
   labels = to_categorical(labels, 13)
   return images, labels


# In[8]:


import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model,model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import cv2, os


# In[15]:


batch_size  = 32
epochs      = 10
num_classes = 13
train_dir   = "out_medium_fix"
size        = 227


# In[10]:


print("Loading Dataset...")

x_train, y_train = get_data(train_dir)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20)

print("x_train shape: {}\ny_train shape: {}\nx_test.shape: {}\ny_test.shape: {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))


# In[16]:


print("\nCreating convolutional neural network...")
model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11,11), input_shape=(224, 224, 3), padding="valid", strides=(4,4)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(4096))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(1000))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(13))
model.add(Activation('softmax'))


# In[17]:


model.summary()


# In[21]:


print("Compiling model...")
opt = keras.optimizers.Adam(lr=0.003)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])


# In[22]:


print("Training model...")

model.fit(x_train, y_train, batch_size=batch_size, validation_split=.25, epochs=epochs)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
print("Saving the model...")


# In[ ]:





# In[ ]:




