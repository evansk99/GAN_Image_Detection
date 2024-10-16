import keras
import tensorflow as tf
from keras import layers, models, losses, optimizers, Input
import pathlib
import pandas as pd
import numpy as np
from Dataset_Generator import *

## Load Model
FILE=r'F:\Model\C64_5.hdf5'
model=tf.keras.models.load_model(FILE) #(filepath=FILE)

##Test Images
number_of_images=48
PHOTO_folder = pathlib.Path(r'F:\IMAGES\MADER\GRAPHICS')
GRAPHICS_folder = pathlib.Path(r'F:\IMAGES\MADER\PHOTOS')

dataset=Dataset_Generator(number_of_images,PHOTO_folder,GRAPHICS_folder)

#Create Test Dataset
df = pd.DataFrame(dataset)
df['label'] = pd.factorize(df['class'])[0]
X = df['path']
Y = df['label']

### TESTING ###
#Constants
block_size=8
block_size2=8
image_size=512

#Variables
number_of_coef_kept=64 #depending on the preprocessing
BATCH_SIZE=4
test_batches=int(number_of_images/BATCH_SIZE)


print("*** Testing Started ***")

x_test = np.zeros((BATCH_SIZE,int(image_size/block_size),int(number_of_coef_kept*image_size/block_size2),3), dtype=np.float16 )
y_test = np.zeros((BATCH_SIZE,1),dtype=np.int16)

## Calculate Confusion Matrix using Test Data ##
TP=0 #True Positive
TN=0 #True Negative
FP=0 #False Positive
FN=0 #False Negative
for j in range(test_batches):
    for i in range(BATCH_SIZE):
      x_test[i,:,:,:]=np.load(X[i+j*BATCH_SIZE])
      y_test[i]=Y[i+j*BATCH_SIZE]

    pred=model.predict(x_test,batch_size=BATCH_SIZE,verbose=0)
    prediction=np.rint(pred)

    #Find the positions where the arrays have the same value or different values (Label-Prediction match or missmatch)
    same_positions = np.where(y_test == prediction)[0]
    diff_positions = np.where(y_test != prediction)[0]

    # Count the number of occurrences of 0 and 1 at the same positions
    TN = TN + np.count_nonzero(y_test[same_positions] == 0)
    TP = TP + np.count_nonzero(y_test[same_positions] == 1)

    # Count the number of occurrences of 0 and 1 at the different positions
    FP = FP + np.count_nonzero(y_test[diff_positions] == 0)
    FN = FN + np.count_nonzero(y_test[diff_positions] == 1)

Total= TP+TN+FP+FN

## Stats ##
print("Accuracy:", float((TP+TN)/Total))
print("Precision:", float(TP/(TP+FP)))
print("Recal:", float(TP/(TP+FN)))
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TOTAL:", TP+TN+FP+FN)
  
