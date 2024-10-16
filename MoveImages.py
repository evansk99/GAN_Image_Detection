import os
import random
import shutil

###
#This funcion moves a specific number of images between a source and a destination folder
###

def move_random_images(source_folder, destination_folder, num_images):
    # Get the list of all files in the source folder
    files = os.listdir(source_folder)

    # Randomly select the specified number of images
    selected_images = random.sample(files, num_images)

    # Move the selected images to the destination folder
    for image in selected_images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.move(source_path, destination_path)
        print(f"Moved image: {image}")

# Example usage
source_folder = r'F:\Dataset\PHOTOS'
destination_folder = r'F:\Dataset\TRAIN\PHOTOS'
num_images_to_move = 5000

move_random_images(source_folder, destination_folder, num_images_to_move)
