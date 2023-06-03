import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from utils import *

def train(model, dataloader, criterion, optimizer, device = 'cuda'):
    model.train()
    total_loss = 0
    total_samples = 0
    total_classification_accuracy = 0
    total_cancer_detection_accuracy = 0
    total_classification_f1 = 0
    total_cancer_detection_f1 = 0
    total_segmentation_iou = 0
    total_segmentation_dice = 0

    for batch in dataloader:
        # Move the batch to the device (e.g., GPU) if available
        batch = [[item.to(device) for item in sub_batch] for sub_batch in batch]

        # Unpack the batch
        images_classification, labels_classification = batch[0]
        images_cancer_detection, labels_cancer_detection = batch[1]
        images_segmentation, masks_segmentation = batch[2]

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        classification_output, cancer_detection_output, segmentation_output = model(images_classification, images_cancer_detection, images_segmentation)

        # Prepare the targets
        targets = {
            'COVID_classification': labels_classification,
            'lung_cancer_detection': labels_cancer_detection,
            'lung_segmentation': masks_segmentation
        }

        # Compute the loss
        loss, outputs = criterion(
            {'COVID_classification': classification_output, 'lung_cancer_detection': cancer_detection_output, 'lung_segmentation': segmentation_output},
            targets
        )

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Update the total loss and total samples
        total_loss += loss.item() * images_classification.size(0)
        total_samples += images_classification.size(0)

        # Compute classification accuracy and F1 score
        classification_preds = classification_output
        cancer_detection_preds = cancer_detection_output

        classification_accuracy = accuracy_score(labels_classification.cpu(), classification_preds.argmax(dim=1).cpu())
        cancer_detection_accuracy = accuracy_score(labels_cancer_detection.cpu(), cancer_detection_preds.argmax(dim=1).cpu())
        classification_f1 = f1_score(labels_classification.cpu(), classification_preds.argmax(dim=1).cpu(), average='weighted')
        cancer_detection_f1 = f1_score(labels_cancer_detection.cpu(), cancer_detection_preds.argmax(dim=1).cpu(), average='weighted')

        total_classification_accuracy += classification_accuracy * images_classification.size(0)
        total_cancer_detection_accuracy += cancer_detection_accuracy * images_classification.size(0)
        total_classification_f1 += classification_f1 * images_classification.size(0)
        total_cancer_detection_f1 += cancer_detection_f1 * images_classification.size(0)

        # Compute segmentation IoU and Dice coefficient
        segmentation_preds = segmentation_output
        segmentation_iou = iou_score(segmentation_preds, masks_segmentation)
        segmentation_dice = dice_coefficient(segmentation_preds, masks_segmentation)

        total_segmentation_iou += segmentation_iou * images_classification.size(0)
        total_segmentation_dice += segmentation_dice * images_classification.size(0)


    # Calculate the average metrics
    average_loss = total_loss / total_samples
    average_classification_accuracy = total_classification_accuracy / total_samples
    average_cancer_detection_accuracy = total_cancer_detection_accuracy / total_samples
    average_classification_f1 = total_classification_f1 / total_samples
    average_cancer_detection_f1 = total_cancer_detection_f1 / total_samples
    average_segmentation_iou = total_segmentation_iou / total_samples
    average_segmentation_dice = total_segmentation_dice / total_samples

    return average_loss, average_classification_accuracy, average_cancer_detection_accuracy, average_classification_f1, average_cancer_detection_f1, average_segmentation_iou, average_segmentation_dice

def evaluate(model, dataloader, criterion, device = 'cuda'):
    model.eval()
    total_loss = 0
    total_samples = 0
    total_classification_accuracy = 0
    total_cancer_detection_accuracy = 0
    total_classification_f1 = 0
    total_cancer_detection_f1 = 0
    total_segmentation_iou = 0
    total_segmentation_dice = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move the batch to the device (e.g., GPU) if available
            batch = [[item.to(device) for item in sub_batch] for sub_batch in batch]

            # Unpack the batch
            images_classification, labels_classification = batch[0]
            images_cancer_detection, labels_cancer_detection = batch[1]
            images_segmentation, masks_segmentation = batch[2]

            # Forward pass
            classification_output, cancer_detection_output, segmentation_output = model(images_classification, images_cancer_detection, images_segmentation)

            # Prepare the targets
            targets = {
                'COVID_classification': labels_classification,
                'lung_cancer_detection': labels_cancer_detection,
                'lung_segmentation': masks_segmentation
            }

            # Compute the loss
            loss, outputs = criterion(
                {'COVID_classification': classification_output, 'lung_cancer_detection': cancer_detection_output, 'lung_segmentation': segmentation_output},
                targets
            )

            # Update the total loss and total samples
            total_loss += loss.item() * images_classification.size(0)
            total_samples += images_classification.size(0)

            # Compute classification accuracy and F1 score
            classification_preds = outputs['COVID_classification']
            cancer_detection_preds = outputs['lung_cancer_detection']
            classification_accuracy = accuracy_score(labels_classification.cpu(), classification_preds.argmax(dim=1).cpu())
            cancer_detection_accuracy = accuracy_score(labels_cancer_detection.cpu(), cancer_detection_preds.argmax(dim=1).cpu())
            classification_f1 = f1_score(labels_classification.cpu(), classification_preds.argmax(dim=1).cpu(), average='weighted')
            cancer_detection_f1 = f1_score(labels_cancer_detection.cpu(), cancer_detection_preds.argmax(dim=1).cpu(), average='weighted')

            total_classification_accuracy += classification_accuracy * images_classification.size(0)
            total_cancer_detection_accuracy += cancer_detection_accuracy * images_classification.size(0)
            total_classification_f1 += classification_f1 * images_classification.size(0)
            total_cancer_detection_f1 += cancer_detection_f1 * images_classification.size(0)

            # Compute segmentation IoU and Dice coefficient
            segmentation_preds = outputs['lung_segmentation']
            segmentation_iou = iou_score(segmentation_preds, masks_segmentation)
            segmentation_dice = dice_coefficient(segmentation_preds, masks_segmentation)

            total_segmentation_iou += segmentation_iou * images_classification.size(0)
            total_segmentation_dice += segmentation_dice * images_classification.size(0)

    # Calculate the average metrics
    average_loss = total_loss / total_samples
    average_classification_accuracy = total_classification_accuracy / total_samples
    average_cancer_detection_accuracy = total_cancer_detection_accuracy / total_samples
    average_classification_f1 = total_classification_f1 / total_samples
    average_cancer_detection_f1 = total_cancer_detection_f1 / total_samples
    average_segmentation_iou = total_segmentation_iou / total_samples
    average_segmentation_dice = total_segmentation_dice / total_samples

    return average_loss, average_classification_accuracy, average_cancer_detection_accuracy, average_classification_f1, average_cancer_detection_f1, average_segmentation_iou, average_segmentation_dice
