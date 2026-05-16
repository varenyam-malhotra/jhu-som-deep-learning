# Varenyam Malhotra
# March 10, 2026

import numpy as np
import tifffile as tiff
import os

def load_mask(path):
    # Load TIFF image
    img = tiff.imread(path)

    # Convert to binary mask (if values are not already 0/1)
    mask = (img > 0).astype(np.uint8)
    return mask

# Computing metric via IOU
def compute_iou(pred_mask, true_mask):
    intersection = np.sum((pred_mask == 1) & (true_mask == 1))
    union = np.sum((pred_mask == 1) | (true_mask == 1))

    if union == 0:
        return 1.0

    return intersection / union

# Computing Metric via Dice
def compute_dice(pred_mask, true_mask):
    intersection = np.sum((pred_mask == 1) & (true_mask == 1))

    total = np.sum(pred_mask) + np.sum(true_mask)

    if total == 0:
        return 1.0

    return (2 * intersection) / total

# Evaluating by loading the paths and calling iou and dice compute methods
def evaluate_masks(pred_path, true_path):
    pred_mask = load_mask(pred_path)
    true_mask = load_mask(true_path)

    if pred_mask.shape != true_mask.shape:
        raise ValueError("Masks must have the same dimensions.")

    iou = compute_iou(pred_mask, true_mask)
    dice = compute_dice(pred_mask, true_mask)

    return iou, dice

# iou, dice = evaluate_masks(pred_file, true_file)

# ----- new loop for all files ------

pred_folder = "..." # place the path to the folder with your prediction TIFF files here
mask_folder = "..." # place the path to the folder with your mask TIFF files here

# sorting the loaded folders via os
# pred_files = sorted(os.listdir(pred_folder))
# mask_files = sorted(os.listdir(mask_folder))

# New idea
pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".tif")])
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".tif")])

iou_scores = []
dice_scores = []

# creating arrays and adding values to array after computing the iou scores and dice scores for each corresponding mask and prediction image
for pred_file, mask_file in zip(pred_files, mask_files):

    pred_path = os.path.join(pred_folder, pred_file)
    mask_path = os.path.join(mask_folder, mask_file)

    iou, dice = evaluate_masks(pred_path, mask_path)

    iou_scores.append(iou)
    dice_scores.append(dice)

# ------- final dataset metrics --------

# calculating mean and then printing data metrics
mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)

print("Total images evaluated:", len(iou_scores))
print("Average IoU Score:", mean_iou)
print("Average Dice Score:", mean_dice)

# print("IoU Score:", iou)
# print("Dice Score:", dice)