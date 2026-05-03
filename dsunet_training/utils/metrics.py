"""
Evaluation metrics for lane detection
"""

import torch
import numpy as np


def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU)"""
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient"""
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    
    return dice.item()


def compute_precision_recall(pred, target, threshold=0.5):
    """Compute Precision and Recall"""
    pred = (pred > threshold).float()
    target = target.float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    
    return precision.item(), recall.item()


def compute_f1(precision, recall):
    """Compute F1 score from precision and recall"""
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1


def compute_metrics(pred, target, threshold=0.5):
    """Compute all metrics"""
    iou = compute_iou(pred, target, threshold)
    dice = compute_dice(pred, target, threshold)
    precision, recall = compute_precision_recall(pred, target, threshold)
    f1 = compute_f1(precision, recall)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    # Test metrics
    pred = torch.rand(4, 1, 480, 640)
    target = torch.randint(0, 2, (4, 1, 480, 640)).float()
    
    metrics = compute_metrics(pred, target)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
