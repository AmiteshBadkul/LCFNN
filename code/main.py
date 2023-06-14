import argparse
import torch
import json
import time
from torch.utils.data import DataLoader

## own work
from trainer import train, evaluate
from dataset import *
from model import *
from loss_fn import *
from utils import *

IN_CHANNELS = 3

def main(args):

    checkpoint_dir = set_up_exp_folder(args.result_path)
    args.results_folder = checkpoint_dir

    with open(checkpoint_dir + 'config.json', 'w') as f:
        json.dump(vars(args), f)

    print('loading ds')
    covid_classification_root = '../datasets/COVID19Classification'
    lung_cancer_detection_root = '../datasets/LungCancerDetection'
    lung_segmentation_image_dir = '../datasets/LungSegmentation/CXR_png/common_images/'
    lung_segmentation_mask_dir = '../datasets/LungSegmentation/masks/common_masks/'
    lung_cancer_metadata_file = '../datasets/LungCancerDetection/jsrt_metadata.csv'

    # Create the datasets
    covid_classification_dataset = COVID19ClassificationDataset(covid_classification_root, (args.image_size, args.image_size))
    lung_cancer_detection_dataset = LungCancerDetectionDataset(lung_cancer_detection_root, lung_cancer_metadata_file, (args.image_size, args.image_size))
    lung_segmentation_dataset = LungSegmentationDataset(lung_segmentation_image_dir, lung_segmentation_mask_dir, (args.image_size, args.image_size))

    # Load datasets
    classification_dataloader = DataLoader(covid_classification_dataset, batch_size=args.batch_size, shuffle=True)
    cancer_detection_dataloader = DataLoader(lung_cancer_detection_dataset, batch_size=args.batch_size, shuffle=True)
    segmentation_dataloader = DataLoader(lung_segmentation_dataset, batch_size=args.batch_size, shuffle=True)

    print('train metrics started')
    # Define dictionaries or dataframes to store evaluation metrics
    train_metrics = {'Loss':[], 'COVID19 Classification Accuracy':[],
                     'Cancer Detection Accuracy':[], 'COVID19 Classification F1':[],
                     'Cancer Detection F1':[], 'Segmentation IoU':[], 'Segmentation Dice Coeff':[]}
    print('model loaded')
    # Create model
    model = MultiTaskModel(
        input_height=args.image_size,
        input_width=args.image_size,
        in_channels=IN_CHANNELS,  # Assuming RGB images
        num_classes_classification=len(covid_classification_dataset.class_labels),
        num_classes_cancer=len(lung_cancer_detection_dataset.diagnosis_labels)
    )

    model.to(args.device)

    task_names = ['COVID_classification', 'lung_cancer_detection', 'lung_segmentation']

    # Define loss function
    criterion = MultiTaskLoss(task_names, weighting_strategy=args.weighting_strategy)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    # # TODO: ADD LEARNING RATE SCHEDULAR LATER

    # Training loop
    for epoch in range(args.num_epochs):
        # Training
        start_time = time.time()
        train_loss, train_covid_accuracy, train_cancer_accuracy, train_covid_f1, train_cancer_f1, train_seg_iou, train_seg_dice = train(model, zip(classification_dataloader, cancer_detection_dataloader,
                                                segmentation_dataloader), criterion, optimizer, args.device)
        print('train_loss', train_loss)
        print('train_covid_accuracy', train_covid_accuracy)
        print('train_cancer_accuracy', train_cancer_accuracy)
        print('train_covid_f1', train_covid_f1)
        print('train_cancer_f1', train_cancer_f1)
        print('train_seg_iou', train_seg_iou)
        print('train_seg_dice', train_seg_dice)

        # Measure GPU memory usage
        gpu_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to megabytes
        print('\n')
        end_time = time.time()
        epoch_time = end_time - start_time
        print('time taken for the epoch.....', epoch_time)
        print('GPU memory usage.....', gpu_memory_usage)

        train_metrics['Loss'].append(train_loss)
        train_metrics['COVID19 Classification Accuracy'].append(train_covid_accuracy)
        train_metrics['Cancer Detection Accuracy'].append(train_cancer_accuracy)
        train_metrics['COVID19 Classification F1'].append(train_covid_f1)
        train_metrics['Cancer Detection F1'].append(train_cancer_f1)
        train_metrics['Segmentation IoU'].append(train_seg_iou)
        train_metrics['Segmentation Dice Coeff'].append(train_seg_dice)


        # Evaluation
        # Perform evaluation after training full model

        # Print progress
        print(f'Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f} - Train Acc COVID19: {train_covid_accuracy:.4f} \nTrain Acc Cancer: {train_cancer_accuracy:.4f} - Train Segmentation IoU: {train_seg_iou:.4f}')
        print('saving metrics')
        df_new = pd.DataFrame(train_metrics)
        df_new.to_csv(args.results_folder + 'metrics.csv')

    df_new = pd.DataFrame(train_metrics)
    # Perform further actions with the metrics or save them to a file
    df_new.to_csv(args.results_folder + 'metrics.csv')

if __name__ == '__main__':
    print('training is starting.......')
    # Define hyperparameters
    parser = argparse.ArgumentParser(description='Lung-CXR-Focused Neural Net (LCFNN)')
    # for model training
    parser.add_argument('--result_path', type=str, default='../results/exp_all_three_tasks/', help='path to save best model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--weighting_strategy', type=str, default='random', help='Weighting Method - [equal, uncertainty, random, dynamic, reduction]')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for optimizer')
    # for data loader

    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')

    args = parser.parse_args()

    main(args)
