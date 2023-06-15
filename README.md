# Lung Chest X-Ray Focused Neural Networks

This repository contains a multi-task deep learning model for COVID-19 classification, lung cancer detection, and lung segmentation. The model is based on the ResNet-50 backbone and utilizes convolutional neural networks (CNNs) for the individual tasks. The main aim logic behind the same is to ensure

## Requirements

- Python 3.9
- PyTorch 1.12
- torchvision 0.13
- Other dependencies (specified in environment/environment.yaml)

## Usage

1. Clone the repository:

```shell
git clone https://github.com/AmiteshBadkul/LCFNN.git
cd LCFNN/environment/
```

2. Create & activate the conda environment:

```shell
conda env create -f environment.yaml
conda activate multi-task-learning
```

3. Prepare the dataset:

- COVID-19 Classification: Place the dataset in the `classification_dataset` directory. Here is the link to the dataset --> [COVID19](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
- Lung Cancer Detection: Place the dataset in the `cancer_detection_dataset` directory. Here is the link to the dataset --> [Lung Cancer Detection](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt)
- Lung Segmentation: Place the dataset in the `segmentation_dataset` directory. Here is the link to the dataset --> [Lung Segmentation](https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/input)

5. Train the model:
The hyperparameters can be modified through CLI.

```shell
python main.py
```

## Project Structure

The project has the following structure:

- `code/` - Contains the code for the model for training and evaluation.
- `results/` - Contains the results of the trained models as well as the model.
- `analysis/` - Contains jupyter notebooks for analysis of the results obtained.

## Results
The results currently obtained are baseline results more model improvements will improve the performance further.

The model achieves the following performance on the test set:

- COVID-19 Classification:
  - Accuracy: 81.05%
  - F1 Score: 80.79%

- Lung Cancer Detection:
  - Accuracy: 52.55%
  - F1 Score: 50.42%

- Lung Segmentation:
  - IoU: 0.26
  - Dice Coefficient: NA

## Notes

Some notes and to-do list:
1. Effective and correct implementation of IoU and Dice Coefficient.
2. Weight balancing techniques.
