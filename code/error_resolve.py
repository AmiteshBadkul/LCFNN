# file meant to check and resolve errors
"""
from miseval import evaluate
import torch
pred = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
target = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
iou = evaluate(pred, target, metric="IoU")
dice = evaluate(pred, target, metric="DSC")
print('iou', iou.item())
print('dice', dice.item())
"""
# fore resolving naming issues
"""
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
"""

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

## for testing out the dataset and the dataloader
"""
# Set the root directories and file paths for each dataset
covid_classification_root = '../datasets/COVID19Classification'
lung_cancer_detection_root = '../datasets/LungCancerDetection'
lung_segmentation_image_dir = '../datasets/LungSegmentation/CXR_png/common_images/'
lung_segmentation_mask_dir = '../datasets/LungSegmentation/masks/common_masks/'
lung_cancer_metadata_file = '../datasets/LungCancerDetection/jsrt_metadata.csv'

# Set the desired image size for resizing
image_size = (256, 256)
batch_size = 1

# Create the datasets
covid_classification_dataset = COVID19ClassificationDataset(covid_classification_root, image_size)
lung_cancer_detection_dataset = LungCancerDetectionDataset(lung_cancer_detection_root, lung_cancer_metadata_file, image_size)
lung_segmentation_dataset = LungSegmentationDataset(lung_segmentation_image_dir, lung_segmentation_mask_dir, image_size)

classification_dataloader = DataLoader(covid_classification_dataset, batch_size=batch_size, shuffle=True)
cancer_detection_dataloader = DataLoader(lung_cancer_detection_dataset, batch_size=batch_size, shuffle=True)
segmentation_dataloader = DataLoader(lung_segmentation_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = MultiTaskModel(
    input_height=image_size[0],
    input_width=image_size[1],
    in_channels=3,  # Assuming RGB images
    num_classes_classification=len(covid_classification_dataset.class_labels),
    num_classes_cancer=len(lung_cancer_detection_dataset.diagnosis_labels),
    kernel_size_backbone = 3, stride_backbone = 2
)

import numpy as np
for classification_batch, cancer_detection_batch, segmentation_batch in zip(classification_dataloader, cancer_detection_dataloader, segmentation_dataloader):
    images_classification, labels_classification = classification_batch
    images_cancer_detection, labels_cancer_detection = cancer_detection_batch
    images_segmentation, masks_segmentation = segmentation_batch

    # Perform your multi-task learning operations here
    # ...

    # For example, print the shapes of the batches
    print("Classification Batch - Images:", images_classification.shape)
    print("Classification Batch - Labels:", labels_classification.shape)
    print("Cancer Detection Batch - Images:", images_cancer_detection.shape)
    print("Cancer Detection Batch - Labels:", labels_cancer_detection.shape)
    print("Segmentation Batch - Images:", images_segmentation.shape)
    print("Segmentation Batch - Masks:", masks_segmentation)
    flattened_tensor_preds = masks_segmentation.flatten().cpu().detach()
    unique_values_pred = np.unique(flattened_tensor_preds)
    print('unique_values_masks', unique_values_pred)

    cov_out, cancer_out, seg_out = model(images_classification, images_cancer_detection, images_segmentation)
    print("Classification Batch - Output:", cov_out.shape)
    print("Cancer Detection Batch - Images:", cancer_out.shape)
    print("Segmentation Batch - Images:", seg_out.shape)
    break
"""
