### Essential Libraries ###

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, losses, optimizers, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
import os
import random
import pandas as pd
import pathlib

from Dataset_Generator import Dataset_Generator,getModel,loadModel

###
#Train and validate new or preexisting models while keeping track of their performance over time!
###

### GPU usage ###
physical_devices=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0],enable=True)


### Model ###
#Create Model
INPUT_SHAPE = (128, 8192, 3)  
model=getModel(input_shape=INPUT_SHAPE)

#Load Pretrained Model
#FILE=r'F:\C10\Models\model_C10_10.hdf5'
#model=tf.keras.models.load_model(FILE)


### Build Dataset ###

number_of_images=10000
PHOTO_folder = pathlib.Path(r'F:\PHOTOS')
GRAPHICS_folder = pathlib.Path(r'F:\GRAPHICS')

dataset=Dataset_Generator(number_of_images,PHOTO_folder,GRAPHICS_folder)

split_index = len(dataset)
#80-20 training-validation 
train_ds = dataset[:int(0.4*split_index)]+dataset[int(0.6*split_index):] 
validation_ds = dataset[int(0.4*split_index):int(0.6*split_index)]

#TRAIN DS
df = pd.DataFrame(train_ds)
df['label'] = pd.factorize(df['class'])[0]
X2 = df['path']
y2 = df['label']

#VALIDATION DS
df = pd.DataFrame(validation_ds)
df['label'] = pd.factorize(df['class'])[0]
X3 = df['path']
Y3 = df['label']


### TRAINING ###
#Constants
block_size=8
block_size2=8
image_size=1024

#Variables
number_of_coef_kept=64
BATCH_SIZE=4
EPOCH=20

tr_batches=int(0.8*number_of_images/BATCH_SIZE) 
val_batches=int(0.2*number_of_images) 

print("*** Training Started ***")
print("\n\nNumber of epochs:" + str(EPOCH),"\nNumber of Batches:"+str(tr_batches))

##Preprocessed Images
x_train = np.zeros((BATCH_SIZE,int(image_size/block_size),int(number_of_coef_kept*image_size/block_size2),3), dtype=np.float16 )
y_train = np.zeros((BATCH_SIZE,1),dtype=np.int16)

x_val = np.zeros((1,int(image_size/block_size),int(number_of_coef_kept*image_size/block_size2),3), dtype=np.float16 )
y_val = np.zeros((1,1),dtype=np.int16)

##Raw pixel images without preprocessing
#x_train = np.zeros((BATCH_SIZE,image_size,image_size,3), dtype=np.float16 )
#y_train = np.zeros((BATCH_SIZE,1),dtype=np.int16)

#x_val = np.zeros((1,image_size,image_size,3), dtype=np.float16 )
#y_val = np.zeros((1,1),dtype=np.int16)

##Metrics
acc=[]        #Training Accuracy History
loss=[]       #Training Loss History
val_acc=[]    #Validation Accuracy History
val_loss=[]   #Validation Loss History

## Custom Training and Validation Loop
t=1
for epoch in range(EPOCH): #Number of epochs
  #print("\nEpoch ",epoch+1)
  # Set a common random seed to randomize batch creation on each epoch
  random_seed = pow((epoch+12),2)%110
  shuffle = np.random.RandomState(seed=random_seed).permutation(len(X2))

  # Shuffle both DataFrames
  X = X2.iloc[shuffle].reset_index(drop=True)
  y = y2.iloc[shuffle].reset_index(drop=True)

  ##  TRAINING ##
  error=0 #Epoch training error
  accuracy = 0 #Epoch training accuracy
  for j in range(tr_batches):
    for i in range(BATCH_SIZE):
      x_train[i,:,:,:]=np.load(X[i+j*BATCH_SIZE])
      y_train[i]=y[i+j*BATCH_SIZE]

    history=model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=1, verbose=0)
    error=error+history.history['loss'][0]
    accuracy=accuracy+history.history['accuracy'][0]
    if(j+1 in [2000]): #Choose the batch number on which a model will be saved
      model.save(r"F:\model_" + str(t) + ".hdf5")
      t=t+1 #unique model name

  acc.append(accuracy/tr_batches)
  loss.append(error/tr_batches)

  ##  VALIDATION ##
  v_accuracy=0 #Epoch validation accuracy
  v_loss=0 #Epoch validation error
  for k in range(val_batches):
    x_val[0,:,:,:]=np.load(X3[k])
    y_val[0,0] = Y3[k]
    val_history=model.evaluate(x=x_val,y=y_val,batch_size=1, verbose=0)
    v_accuracy=v_accuracy+val_history[1]
    v_loss=v_loss+val_history[0]

  val_acc.append(v_accuracy/val_batches)
  val_loss.append(v_loss/val_batches)
  #print("Validation Accuracy:",val_acc[-1])
  #print("Validation Loss:",val_loss[-1])

print("TRAINING COMPLETED :D")
