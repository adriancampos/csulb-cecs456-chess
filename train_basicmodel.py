import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import glob, os
from PIL import Image

from sklearn.model_selection import train_test_split


# Define the class names so things look nicer
class_names = [
    "blank",
    "pawn (Black)",
    "knight (Black)",
    "bishop (Black)",
    "rook (Black)",
    "queen (Black)",
    "king (Black)",
    "pawn (White)",
    "knight (White)",
    "bishop (White)",
    "rook (White)",
    "queen (White)",
    "king (White)"    
]
label_indices = [
    "0",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K"
]

# load data from disk
indirectory = "in_split/"

data_features = []
data_labels = []
for infile in glob.glob(indirectory + "/**/*.jp*g", recursive=True):
    boardImg = Image.open(infile)
    data_features.append(np.array(boardImg))
    
    label = os.path.split(infile)[1]
    label = label.split('_')[0]
    data_labels.append(label_indices.index(label))
    
    
data_features = np.array(data_features)
data_labels = np.array(data_labels)

# Assume that there's a label for every image
assert data_features.shape[0] == data_labels.shape[0]

print(data_features.shape)

"""# Preprocess the data"""

# Display the first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_features[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[data_labels[i]])         
plt.show()

label_names = []
# Load the class name for every label
for i in range(len(data_labels)):
    label_names.append(label_indices[data_labels[i]])

# Scale the dataset to 0 to 1
data_features = data_features / 255.0

# Split the dataset up
data_train, data_test, labels_train, labels_test = train_test_split(data_features, data_labels, test_size=0.20)
print("train:",data_train.shape)
print("test:",data_test.shape)


del data_features
del data_labels

"""# Build The Model"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50,50,3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(13, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

"""# Train"""

# Train the model with vadidation
model.fit(data_train, labels_train, validation_split=.20, epochs=5)
model.save('model.h5')

# Evaluate accuracy
test_loss, test_acc = model.evaluate(data_test, labels_test)
print('Test accuracy:', test_acc)

"""# Play with the results"""

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(13), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
    
predictions = model.predict(data_test)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 10
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, labels_test, data_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, labels_test)
plt.show()