# https://www.tensorflow.org/tutorials/quickstart/beginner

# taken from the TensorFlow web page
# first example, MNIST digit dataset

import tensorflow as tf
from matplotlib.image import imread
import os
import numpy as np
import sys


print("Usage: python mnist.py /path/to/train/data /path/to/test/data")

smotedatapath = sys.argv[1]
mnistdatapath = sys.argv[2]


traindata = []
trainlabel = []

print("loading train images by label")
for currentlabel in range(0,10):
  print(str(currentlabel))
  currentlabelpath = smotedatapath + '/' + str(currentlabel)
  for currentfilename in os.listdir(currentlabelpath):
    currentpath = currentlabelpath + '/' + currentfilename
#    print(currentpath)
    currentimage = np.ndarray.tolist(imread(currentpath))
    traindata.append(currentimage)
    trainlabel.append(currentlabel)
#turn into numpy arrays
traindata = np.array(traindata).astype(np.float32)
trainlabel = np.array(trainlabel).astype(np.float32)






testdata = []
testlabel = []

print("loading test images by label")
for currentlabel in range(0,10):
  print(str(currentlabel))
  currentlabelpath = mnistdatapath + '/' + str(currentlabel)
  for currentfilename in os.listdir(currentlabelpath):
    currentpath = currentlabelpath + '/' + currentfilename
#    print(currentpath)
    currentimage = np.ndarray.tolist(imread(currentpath))
    testdata.append(currentimage)
    testlabel.append(currentlabel)
#turn into numpy arrays
testdata = np.array(testdata).astype(np.float32)
testlabel = np.array(testlabel).astype(np.float32)


print("\n\n\n\n\n")
print(traindata.shape)
print(trainlabel.shape)
print(testdata.shape)
print(testlabel.shape)
print("\n\n\n\n\n")









model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(traindata, trainlabel, verbose=2, epochs=100)

  
  
a=model.evaluate(testdata, testlabel)
print(a)
