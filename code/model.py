import torch.nn as nn
from torchvision import models
import torch

## adapted work
from resnet import resnetsimple, resnet18, resnet34

## Baseline Model
class ResNetBackbone(nn.Module):
    def __init__(self, input_height, input_width, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResNetBackbone, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        for param in self.resnet.parameters():
            param.requires_grad = False

        # Remove the fully connected layer (head) from the ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc layers

        # Add batch normalization and activation function after each convolutional layer
        self.resnet = nn.Sequential(
            self.resnet,
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        # Add an additional convolutional layer with variable kernel size and stride
        self.conv = nn.Conv2d(2048, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size//2)

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv(x)
        return x

class COVIDClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(COVIDClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class LungCancerDetectionHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LungCancerDetectionHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class LungSegmentationModel(nn.Module):
    def __init__(self, input_height, input_width, in_channels, num_classes = 1):
        super(LungSegmentationModel, self).__init__()
        self.upsample = nn.Upsample(size = (input_height, input_width), mode = 'bilinear', align_corners = True)
        self.conv = nn.Conv2d(2048, num_classes, kernel_size = 1)
        self.activation = nn.Sigmoid()
        self.threshold = 0.5

    def forward(self, x):
        upsampled_features = self.upsample(x)
        mask = self.conv(upsampled_features)
        mask = self.activation(mask)
        binary_mask = (mask > self.threshold).float()
        return binary_mask.float()

class LungSegmentationModel2(nn.Module):
    """
    This model isn't helpful. It gives negative performance
    """
    def __init__(self, input_height, input_width, in_channels, num_classes=1):
        super(LungSegmentationModel2, self).__init__()
        self.upsample = nn.Upsample(size=(input_height, input_width), mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        upsampled_features = self.upsample(x)
        x = self.conv1(upsampled_features)
        x = self.conv2(x)
        mask = self.conv3(x)
        mask = self.activation(mask)
        return mask

## Second Version
## This contains the resnet encoder based input
class ResNetModularBackbone(nn.Module):
    def __init__(self, input_height, input_width, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResNetModularBackbone, self).__init__()
        self.backbone = resnet18(in_channels)
        self.input_height = input_height
        self.input_width = input_width
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Modify the last convolutional layer based on the specified output channels
        self.backbone.blocks[-1].blocks[-1] = nn.Conv2d(
            self.backbone.blocks[-1].blocks[-1].expanded_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )

        # Add an additional convolutional layer with variable kernel size and stride
        self.conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, input_height, input_width, in_channels, kernel_size_backbone, stride_backbone, num_classes_classification, num_classes_cancer):
        super(MultiTaskModel, self).__init__()
        self.backbone = ResNetModularBackbone(input_height, input_width, in_channels, out_channels=2048, kernel_size = kernel_size_backbone, stride = stride_backbone)
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
