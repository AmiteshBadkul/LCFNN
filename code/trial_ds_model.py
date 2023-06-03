from dataset import *
from trainer import *
from model import *
from loss_fn import *

import os
import torch
import pandas as pd

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
    num_classes_cancer=len(lung_cancer_detection_dataset.diagnosis_labels)
)

print(model)

learning_rate = 0.001
weight_decay = 0.0001

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

EPOCHS = 5
task_names = ['COVID_classification', 'lung_cancer_detection', 'lung_segmentation']
loss_function = MultiTaskLoss(task_names, weighting_strategy='random')

for epoch in range(EPOCHS):
    print(epoch)
    r1 = train(model, zip(classification_dataloader, cancer_detection_dataloader, segmentation_dataloader),
               loss_function, optimizer, device = 'cuda')
    r2 = evaluate(model, zip(classification_dataloader, cancer_detection_dataloader, segmentation_dataloader),
               loss_function, device = 'cuda')
    print(r2)
    break
