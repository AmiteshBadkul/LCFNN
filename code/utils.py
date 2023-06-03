from datetime import datetime
import torch
import os

def set_up_exp_folder(path):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print('timestamp: ',timestamp)
    save_folder = path
    if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
    checkpoint_dir = '{}/exp{}/'.format(save_folder, timestamp)
    if os.path.exists(checkpoint_dir) == False:
            os.mkdir(checkpoint_dir)
    return checkpoint_dir

def iou_score(pred, target):
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    iou = intersection / (union + 1e-7)  # Adding a small epsilon to avoid division by zero
    return iou.float().item()

def dice_coefficient(pred, target):
    intersection = torch.logical_and(pred, target).sum()
    dice = (2 * intersection) / (pred.sum() + target.sum() + 1e-7)
    return dice.float().item()
