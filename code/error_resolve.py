# file meant to check and resolve errors

import os
import shutil

image_dir = '../datasets/LungSegmentation/CXR_png'
mask_dir = '../datasets/LungSegmentation/masks'

# Create output directories for common images and masks
common_images_dir = os.path.join(image_dir, 'common_images')
common_masks_dir = os.path.join(mask_dir, 'common_masks')
os.makedirs(common_images_dir, exist_ok=True)
os.makedirs(common_masks_dir, exist_ok=True)

# Get the list of image and mask file names
image_files = os.listdir(image_dir)
mask_files = os.listdir(mask_dir)

# Extract the base file names without extensions
image_names = set([os.path.splitext(file)[0] for file in image_files])
mask_names = set([os.path.splitext(file)[0].replace('_mask', '') for file in mask_files])

# Find the common pairs present in both image and mask directories
common_pairs = image_names.intersection(mask_names)

# Move the common pairs to the common_images and common_masks directories
for pair in common_pairs:
    image_path = os.path.join(image_dir, pair + '.png')
    mask_path = os.path.join(mask_dir, pair + '_mask.png')
    mask_path = mask_path.replace("\\", "/")  # Replace backslash with forward slash

    common_image_path = os.path.join(common_images_dir, pair + '.png')
    common_mask_path = os.path.join(common_masks_dir, pair + '_mask.png')

    shutil.copy(image_path, common_image_path)
    shutil.copy(mask_path, common_mask_path)

print("Common pairs moved to 'common_images' and 'common_masks' directories.")


## for renaming the masks files
"""
import os

directory = '../datasets/LungSegmentation/masks'  # Replace with the path to your directory

# Get the list of files in the directory
files = os.listdir(directory)

# Iterate over the files and relabel them if necessary
for filename in files:
    if '_mask' not in filename:
        new_filename = filename.replace('.png', '_mask.png')
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed file: {filename} -> {new_filename}")
"""
