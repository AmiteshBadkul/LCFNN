import torch.nn as nn
from torchvision import models
import torch

class ResNetBackbone(nn.Module):
    def __init__(self, input_height, input_width, in_channels, out_channels):
        super(ResNetBackbone, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.out_channels = out_channels

        for param in self.resnet.parameters():
            param.requires_grad = False

        # Remove the fully connected layer (head) from the ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc layers

        # Add an additional convolutional layer for obtaining feature maps
        self.conv = nn.Conv2d(2048, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv(x)
        return x

class COVIDClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(COVIDClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(input_size, num_classes)
        #self.fc2 = nn.Linear(num_classes, 1)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.fc2(x)

        return x

class LungCancerDetectionHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LungCancerDetectionHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(input_size, num_classes)
        #self.fc2 = nn.Linear(num_classes, 1)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.fc2(x)
        return x

class LungSegmentationModel(nn.Module):
    def __init__(self, input_height, input_width, in_channels, num_classes = 1):
        super(LungSegmentationModel, self).__init__()
        self.upsample = nn.Upsample(size=(input_height, input_width), mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        upsampled_features = self.upsample(x)
        mask = self.conv(upsampled_features)
        #threshold = 0.5
        #binary_mask = (mask > threshold).float()
        mask = self.activation(mask)
        return mask

class MultiTaskModel(nn.Module):
    def __init__(self, input_height, input_width, in_channels, num_classes_classification, num_classes_cancer):
        super(MultiTaskModel, self).__init__()
        self.backbone = ResNetBackbone(input_height, input_width, in_channels, out_channels=2048)
        self.classification_head = COVIDClassificationHead(2048, num_classes_classification)
        self.cancer_detection_head = LungCancerDetectionHead(2048, num_classes_cancer)
        self.segmentation_model = LungSegmentationModel(input_height, input_width, in_channels)

    def forward(self, classification_input, cancer_detection_input, segmentation_input):
        classification_features = self.backbone(classification_input)
        cancer_detection_features = self.backbone(cancer_detection_input)
        segmentation_features = self.backbone(segmentation_input)

        classification_output = self.classification_head(classification_features)
        cancer_detection_output = self.cancer_detection_head(cancer_detection_features)
        segmentation_output = self.segmentation_model(segmentation_features)

        return classification_output, cancer_detection_output, segmentation_output
