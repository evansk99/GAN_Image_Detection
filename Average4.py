import os
import shutil
from DCT import *

###
#This function downsizes the images by splitting the pixels of the image in groups of 4 and
#assigning a single value to the group which is the average intensity
###

def average_downsize(img_array):
    
    new_width = 512
    new_height = 512
    
    resized_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for x in range(new_width):
        for y in range(new_height):
            pixel1 = img_array[y * 2, x * 2]
            pixel2 = img_array[y * 2, x * 2 + 1]
            pixel3 = img_array[y * 2 + 1, x * 2]
            pixel4 = img_array[y * 2 + 1, x * 2 + 1]
            
            avg_pixel = (pixel1 + pixel2 + pixel3 + pixel4) // 4
            
            resized_array[y, x] = avg_pixel
    return resized_array

###
#This procedure shrinks the images using the afformentioned method
#while also carrying out the preprocessing stage of the proposed method
###

# Specify the paths for the input and output directories
input_folder = r'F:\IMAGES\Dataset\TRAIN\PHOTOS'
output_folder = r'F:\IMAGES\average4\TRAIN\PHOTOS'

new_width = 512
new_height = 512

# Set the number of images to select
num_images = 2

# Loop over each image file in the input folder
count=0
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
        image_path = os.path.join(input_folder, filename)

        image = np.array(Image.open(image_path))
        image2 = average_downsize(image)

        final=dct_3d(image2)  

        file_name = f"/{count}.npy"  

        # Save the modified image to the output folder
        output_path = os.path.join(output_folder+file_name)
        np.save(output_path, final)

        count=count+1

    if(count==num_images):
      break

print("All images processed!")
