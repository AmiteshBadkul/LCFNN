import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize

# only for testing out the architecture
from model import MultiTaskModel

class COVID19ClassificationDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.class_labels = {'COVID19': 0, 'NORMAL': 1, 'PNEUMONIA': 2}
        self.image_paths = self._load_image_paths()
        self.transform = Resize(image_size)

    def _load_image_paths(self):
        image_paths = []
        for class_label in self.class_labels.keys():
            class_path = os.path.join(self.root_dir, class_label)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                image_paths.append((file_path, self.class_labels[class_label]))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_path, class_label = self.image_paths[idx]
        image = Image.open(file_path).convert('RGB')
        image = self.transform(image)
        image = ToTensor()(image)
        return image, class_label

class LungCancerDetectionDataset(Dataset):
    def __init__(self, root_dir, metadata_file, image_size):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = Resize(image_size)
        self.diagnosis_labels = {'benign': 0, 'malignant': 1, 'non-nodule': 2}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        study_id = self.metadata.loc[idx, 'study_id']
        file_path = os.path.join(self.root_dir, 'images', study_id)
        image = Image.open(file_path).convert('RGB')
        image = self.transform(image)
        image = ToTensor()(image)

        diagnosis = self.metadata.loc[idx, 'state']
        label = self.diagnosis_labels[diagnosis]  # Convert diagnosis to numerical label

        return image, label

class LungSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = os.listdir(image_dir)
        self.transform = Resize(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, file_name)
        image_path = image_path.replace("\\", "/")

        base_filename = os.path.splitext(file_name)[0]
        mask_filename = base_filename + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)
        mask_path = mask_path.replace("\\", "/")

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = ToTensor()(image)

        mask = Image.open(mask_path).convert('L')
        mask = self.transform(mask)
        mask = ToTensor()(mask)

        return image, mask

def create_multitask_dataloader(classification_dataset, cancer_detection_dataset, segmentation_dataset, batch_size):
    classification_dataloader = DataLoader(classification_dataset, batch_size=batch_size, shuffle=True)
    cancer_detection_dataloader = DataLoader(cancer_detection_dataset, batch_size=batch_size, shuffle=True)
    segmentation_dataloader = DataLoader(segmentation_dataset, batch_size=batch_size, shuffle=True)

    combined_dataloader = zip(classification_dataloader, cancer_detection_dataloader, segmentation_dataloader)
    return DataLoader(combined_dataloader, batch_size=1, shuffle=True, collate_fn=collate_fn_combined)

def collate_fn_combined(batch):
    images_classification, labels_classification = zip(*batch[0])
    images_cancer_detection, labels_cancer_detection = zip(*batch[1])
    images_segmentation, masks_segmentation = zip(*batch[2])

    # Combine the batches into a single batch
    combined_batch = (
        torch.stack(images_classification),
        torch.tensor(labels_classification),
        torch.stack(images_cancer_detection),
        torch.tensor(labels_cancer_detection),
        torch.stack(images_segmentation),
        torch.stack(masks_segmentation)
    )

    return combined_batch

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
    num_classes_cancer=len(lung_cancer_detection_dataset.diagnosis_labels)
)


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
    print("Segmentation Batch - Masks:", masks_segmentation.shape)

    cov_out, cancer_out, seg_out = model(images_classification, images_cancer_detection, images_segmentation)
    print("Classification Batch - Output:", cov_out.shape)
    print("Cancer Detection Batch - Images:", cancer_out.shape)
    print("Segmentation Batch - Images:", seg_out.shape)
    break
"""
