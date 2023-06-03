import torch
import torch.nn as nn
import random

class MultiTaskLoss(nn.Module):
    def __init__(self, task_names, weighting_strategy='equal'):
        super(MultiTaskLoss, self).__init__()
        self.task_names = task_names
        self.weighting_strategy = weighting_strategy

    def forward(self, outputs, targets, metric_value=None):
        losses = {}
        total_loss = 0

        loss_weights = self.calculate_loss_weights(outputs, targets, metric_value)

        for task_name in self.task_names:
            if task_name in outputs and task_name in targets:
                loss_func = self.get_loss_function(task_name)
                loss_weight = loss_weights[task_name]
                output = outputs[task_name]
                target = targets[task_name]
                loss = loss_func(output, target)
                losses[task_name] = loss.item()
                total_loss += loss * loss_weight

        return total_loss, losses

    def get_loss_function(self, task_name):
        # Define the loss function for each task
        if task_name == 'COVID_classification':
            return nn.CrossEntropyLoss()
        elif task_name == 'lung_cancer_detection':
            return nn.CrossEntropyLoss()
        elif task_name == 'lung_segmentation':
            return nn.BCEWithLogitsLoss()

    def calculate_loss_weights(self, outputs, targets, metric_value=None):
        loss_weights = {}

        num_tasks = len(self.task_names)
        if num_tasks > 0:
            if self.weighting_strategy == 'equal':
                equal_weight = 1.0 / num_tasks
                loss_weights = {task_name: equal_weight for task_name in self.task_names}

            elif self.weighting_strategy == 'uncertainty':
                loss_weights = self.calculate_uncertainty_weights(outputs, targets)

            elif self.weighting_strategy == 'random':
                loss_weights = {task_name: random.uniform(0.0, 1.0) for task_name in self.task_names}

            elif self.weighting_strategy == 'dynamic':
                loss_weights = self.calculate_dynamic_weights(outputs, targets, metric_value)

            elif self.weighting_strategy == 'reduction':
                loss_weights = self.calculate_reduction_weights()

        return loss_weights

    def calculate_uncertainty_weights(self, outputs, targets):
        uncertainty_weights = {}
        epsilon = 1e-7

        for task_name in self.task_names:
            if task_name in outputs and task_name in targets:
                output = outputs[task_name]
                target = targets[task_name]

                # Calculate uncertainty based on outputs and targets
                # Here, we calculate the mean squared error between the predicted output and the target
                # The uncertainty is inversely proportional to the mean squared error
                uncertainty = torch.mean((output - target) ** 2)
                uncertainty_weights[task_name] = 1.0 / (uncertainty + epsilon)

        # Normalize the uncertainty weights
        total_weight = sum(uncertainty_weights.values())
        uncertainty_weights = {task_name: weight / total_weight for task_name, weight in uncertainty_weights.items()}

        return uncertainty_weights

    def calculate_dynamic_weights(self, outputs, targets, metric_value=None):
        dynamic_weights = {}

        # Update weights dynamically based on task performance and metric value
        # Here, we assume a simple dynamic weight averaging strategy
        for task_name in self.task_names:
            if task_name in outputs and task_name in targets:
                # Update the loss weights based on validation metrics or other criteria
                # Here, we randomly assign weights for demonstration purposes
                dynamic_weights[task_name] = random.uniform(0.0, 1.0) * metric_value

        # Normalize the dynamic weights
        total_weight = sum(dynamic_weights.values())
        dynamic_weights = {task_name: weight / total_weight for task_name, weight in dynamic_weights.items()}

        return dynamic_weights

    def calculate_reduction_weights(self):
        reduction_weights = {}

        # Reduce the number of tasks for ablation studies by setting weights to zero
        # Here, we reduce the number of tasks by setting the weight of 'lung_cancer_detection' to zero
        for task_name in self.task_names:
            reduction_weights[task_name] = 0.0 if task_name == 'lung_cancer_detection' else 1.0

        return reduction_weights
