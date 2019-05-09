#!/usr/bin/env python
# coding: utf-8

# In[46]:


import os
import cv2
import numpy as np
import keras
from random import randrange
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json


# In[19]:


def train_test_split(feature_labels, size, test_size=0.2):
    data = []
    train_size = (1-test_size)*len(feature_labels)
    
    while len(data) < train_size:
        index = randrange(len(feature_labels))
        data.append(feature_labels.pop(index))
    x_train = [x[0] for x in data]
    y_train = [x[1] for x in data]
    x_test = [x[0] for x in feature_labels]
    y_test = [x[1] for x in feature_labels]
    # Reshape and scale the features from 0-255 to 0.0-1.0
    x_train = (np.array(x_train)/255).reshape(-1, size, size, 3)
    x_test = (np.array(x_test)/255).reshape(-1, size, size, 3)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return x_train, x_test, y_train, y_test
    
def create_train_data(train_dir, size=150):
    feature_labels = [] # Initialize empty list of [[feature, label]]
    
    for file in os.listdir(train_dir):
        label = 0 if file.split('.')[0] == 'cat' else 1
        image = cv2.resize(cv2.imread(os.path.join(train_dir, file)), (size, size))
        feature_labels.append([image, label])
        
    return train_test_split(feature_labels, size, test_size=0.2)
    


# In[33]:


BATCH_SIZE     = 64
EPOCHS_COUNT   = 20
NUM_OF_CLASSES = 2
TRAIN_DIR      = "classify-cats-dogs/train"
INPUT_SIZE     = 224


# In[21]:


print("Preparing the dataset...")
x_train, x_test, y_train, y_test = create_train_data(TRAIN_DIR, INPUT_SIZE)


# In[43]:


print("Setting up AlexNet Architecture CNN...")
model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11,11), input_shape=(INPUT_SIZE, INPUT_SIZE, 3), padding="valid", strides=(4,4)))
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
model.add(Dense(2))
model.add(Activation('softmax'))


# In[44]:


print("Compiling model...")
opt = keras.optimizers.Adam(lr=0.00001)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
print("Training model...")
model.fit(x_train, y_train, validation_split=.25, epochs=EPOCHS_COUNT)


# In[45]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Accuracy on test dataset: {}".format(test_acc))
print("Loss on test dataset: {}".format(test_loss))


# In[47]:


model.save_weights("model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:




