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
        #image_path = image_path.replace("\\", "/")

        base_filename = os.path.splitext(file_name)[0]
        mask_filename = base_filename + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)
        #mask_path = mask_path.replace("\\", "/")

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = ToTensor()(image)

        mask = Image.open(mask_path).convert('L')
        mask = self.transform(mask)
        mask = ToTensor()(mask)

        # Apply thresholding to obtain a binary mask
        threshold = 0.5  # Adjust the threshold as needed
        mask = torch.where(mask > threshold, torch.tensor(1), torch.tensor(0))

        return image, mask.float()

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
