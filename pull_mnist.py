#Downloader which pulls all the data from keras's mnist repo to a local directory

import tensorflow as tf
import scipy.misc
from PIL import Image
import os
import sys

if(len(sys.argv) < 2):
  printf("Usage: python3 pull_mnist.py <Full path to location to store data>\n");
  sys.exit()

datasaveloc = sys.argv[1]

mnist = tf.keras.datasets.mnist
data = mnist.load_data()

for i in range(0,len(data[0][0])-1):
  currentdata = data[0][0][i]
  currentlabel = data[0][1][i]
  currentimage = Image.fromarray(currentdata, 'L')
  saveloc = datasaveloc + str(currentlabel) + '/' + str(i) + '.jpg'
  currentimage.save(saveloc,"JPEG")
