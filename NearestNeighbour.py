import os
import shutil
from DCT import *

from PIL import Image
from scipy import ndimage

###
#This function shrinks the input images by applying nearest neighbour interpolation
###

def nearest_neighbor_resize(image, new_shape):
    return ndimage.zoom(image, (new_shape[0] / image.shape[0], new_shape[1] / image.shape[1], 1), order=0)

# Specify the paths for the input and output directories
input_folder = r'F:\IMAGES\Dataset\TRAIN\GRAPHICS'
output_folder = r'D:F:\IMAGES\NNI\TRAIN\GRAPHICS'

new_width = 512
new_height = 512

# Set the number of images to select
num_images = 8000

# Loop over each image file in the input folder
count=0
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter image files

        image_path = os.path.join(input_folder, filename)

        image = np.array(Image.open(image_path))
        image2 = nearest_neighbor_resize(image, (new_height, new_width))

        final=dct_3d(image2)

        file_name = f"/{count}.npy"  

        # Save the modified image to the output folder
        output_path = os.path.join(output_folder+file_name)
        np.save(output_path, final)

        count=count+1

    if(count==num_images):
      break

print("All images processed!")
