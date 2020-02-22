#
# Script used for generating extra data via SMOTE to balance imbalanced datasets
#



from collections import Counter
import numpy as np
from PIL import Image
from keras.preprocessing.image import image
from keras.models import Model
from keras.layers import Input, Dense
import csv
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from scipy import sparse
import os
import sys

if(len(sys.argv) < 2):
  print("Usage: python3 smoteimagedataset.py <Full Path To Dataset 1> <Image Width> <Image Height> <Number of data to generate> <Location to save data to> <neighbor constant> <random constant>")
  sys.exit()


#
# Read in images to linearized numpy array
#

datasetloc = sys.argv[1]
dim_x = int(sys.argv[2])
dim_y = int(sys.argv[3])
num_to_generate = int(sys.argv[4])
saveloc = sys.argv[5]
neighbor_count = int(sys.argv[6])
random_seed = int(sys.argv[7])

x_train = []
y_train = []

print("Reading in dataset")

size1 = 0
dir1 = datasetloc
for imgname in os.listdir(dir1):
  size1 = size1 + 1;
  img = image.load_img(dir1+imgname,target_size=(dim_x,dim_y))
  linearized = []
  for x in range(0,dim_x):
    for y in range(0,dim_y):
      linearized.append(img.getpixel((x,y))[0])
  linearized = np.array(linearized)
  x_train.append(linearized)
  y_train.append(1)

print("read " + str(size1) + " images!")



print("Generating dummy set")

size2 = 0;
for x in range(0,size1+num_to_generate):
  size2 = size2 + 1;
  linearized = []
  for x in range(0,dim_x):
    for y in range(0,dim_y):
      value = 0
      linearized.append(value)
  linearized = np.array(linearized)
  x_train.append(linearized)
  y_train.append(0)

print("Done generating dummy set")
print("Generated " + str(size2) + " dummy images!")


# run SMOTE

print("Running SMOTE with neighbor count " + str(neighbor_count))

sm = SMOTE(random_state=random_seed,  k_neighbors=neighbor_count)
X_res, Y_res =  sm.fit_resample(x_train,y_train)



#
# Print an example image in .jpg format on hard disk
#

print ("Outputting result data")

for i in range(0,num_to_generate):
  arrayloc = 0-i-1
  newdata = X_res[arrayloc]
  imgvals = []
  for x in range(0,dim_x):
    subarray = []
    for y in range(0,dim_y):
      subarray.append(newdata[y*(dim_x)+x])
#      subarray.append(y*5)
    imgvals.append(subarray)
  imgvals = np.array(imgvals)
  generatedimage = Image.fromarray(imgvals.astype('uint8'), 'L')
  generatedimage.save(saveloc + str(i) + ".jpg", "JPEG")
