# Varenyam Malhotra
# March 7 2026

import numpy as np
import tifffile as tiff

def load_mask(path):
    # Load TIFF image
    img = tiff.imread(path)

    # Convert to binary mask (if values are not already 0/1)
    mask = (img > 0).astype(np.uint8)

    return mask


def compute_iou(pred_mask, true_mask):
    intersection = np.sum((pred_mask == 1) & (true_mask == 1))
    union = np.sum((pred_mask == 1) | (true_mask == 1))

    if union == 0:
        return 1.0

    return intersection / union


def compute_dice(pred_mask, true_mask):
    intersection = np.sum((pred_mask == 1) & (true_mask == 1))

    total = np.sum(pred_mask) + np.sum(true_mask)

    if total == 0:
        return 1.0

    return (2 * intersection) / total


def evaluate_masks(pred_path, true_path):
    pred_mask = load_mask(pred_path)
    true_mask = load_mask(true_path)

    if pred_mask.shape != true_mask.shape:
        raise ValueError("Masks must have the same dimensions.")

    iou = compute_iou(pred_mask, true_mask)
    dice = compute_dice(pred_mask, true_mask)

    return iou, dice


# Example usage
pred_file = "pred001 copy.tif"
true_file = "mask001 copy.tif"

iou, dice = evaluate_masks(pred_file, true_file)

print("IoU Score:", iou)
print("Dice Score:", dice)