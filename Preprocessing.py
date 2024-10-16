import os
import shutil
from DCT import *

###
#This procedure carries out the proposed preprocessing by applying the 
#blockwise DCT and rearrangings its' coefficients in a systematic way
###

# Specify the paths for the input and output directories
input_folder = r'F:\IMAGES\LAGOS\GRAPHICS'
output_folder = r'F:\IMAGES\TRAIN\GRAPHICS'

# Set the number of images to select
num_images = 24

# Loop over each image file in the input folder
count=0
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)

        image = np.array(Image.open(image_path))

        final=dct_3d(image)

        file_name = f"/{count}.npy"  

        # Save the modified image to the output folder
        output_path = os.path.join(output_folder+file_name)
        np.save(output_path, final)

        count=count+1

    if(count==num_images):
      break

print("All images processed!")
