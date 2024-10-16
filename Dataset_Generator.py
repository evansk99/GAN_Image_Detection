import pathlib
import os
import random
from random import sample

import tensorflow as tf
import keras
from keras import layers, models, losses, optimizers, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

###
#This function returns a list that contains the paths of the images that are going to be used
#either to train or to test a new or a pretrained neural network model
###

def Dataset_Generator(number_of_images,PHOTO_folder,GRAPHICS_folder):

  num_images = int(number_of_images*0.5)
  
  # Find images of dataset
  dataset = []
    
  # Get a list of all the image filenames in the PHOTO_folder
  files = os.listdir(PHOTO_folder)
  # Choose num_images random unique files from the folder
  rand_files = random.sample(set(files), num_images)

  for file in rand_files:
    sample = {'path': os.path.join(PHOTO_folder, file),'class':0}
    dataset.append(sample)

  #Repeat process for second folder
  files2 = os.listdir(GRAPHICS_folder)
  # Choose num_images random unique files from the folder
  rand_files2 = random.sample(set(files2), num_images)

  for file2 in rand_files2:
    sample = {'path': os.path.join(GRAPHICS_folder, file2),'class':1}
    dataset.append(sample)

  return dataset

###
#This function creates the CNN architecture presented in the thesis
###

def getModel(input_shape):
  ### CNN Architecture ###
  # Define the model architecture
  input = keras.Input(shape=input_shape,name="START")
  x = layers.Conv2D(filters=32,kernel_size=(5, 5),strides=1,padding="same",activation="relu")(input)
  x = layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid")(x)
  x = layers.Conv2D(filters=64,kernel_size=(3, 3),strides=1,padding="same",activation="relu")(x) 
  x = layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid")(x)
  x = layers.Conv2D(filters=64,kernel_size=(3, 3),strides=1,padding="same",activation="relu")(x) 
  x = layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid")(x)
  x = layers.Conv2D(filters=128,kernel_size=(3, 3),strides=1,padding="same",activation="relu")(x) 
  x = layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid")(x)
  x = layers.Conv2D(filters=128,kernel_size=(3, 3),strides=1,padding="same",activation="relu")(x) 
  x = layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid")(x)
  x = keras.layers.Flatten()(x)
  dense = layers.Dense(1, activation="sigmoid")  #or 2 nodes with softmax activation and categorical cross-entropy loss
  output = dense(x)

  model = keras.Model(inputs=input, outputs=output, name="test_model")
  model.summary()

  ### Model Compilation ###
  opt = keras.optimizers.RMSprop(learning_rate=0.001)
  model.compile(
  optimizer=opt,  #optimization algorithm
  loss=tf.keras.losses.binary_crossentropy, #loss function
  metrics=['accuracy'])
  
  return model

###
#This function loads a pretrained model for further training or testing
###

def loadModel(filepath):

  model=tf.keras.models.load_model(filepath)
  
  ### Model Compilation ###
  opt = keras.optimizers.RMSprop(learning_rate=0.001)
  model.compile(
  optimizer=opt, 
  loss=tf.keras.losses.binary_crossentropy,
  metrics=['accuracy'])
  
  return model
