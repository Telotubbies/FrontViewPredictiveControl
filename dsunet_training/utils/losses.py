"""
Loss functions for lane detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + BCE + Focal Loss"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total_loss = (
            self.dice_weight * dice +
            self.bce_weight * bce +
            self.focal_weight * focal
        )
        
        return total_loss, {
            'dice': dice.item(),
            'bce': bce.item(),
            'focal': focal.item(),
            'total': total_loss.item()
        }


if __name__ == "__main__":
    # Test losses
    pred = torch.randn(4, 1, 480, 640)
    target = torch.randint(0, 2, (4, 1, 480, 640)).float()
    
    criterion = CombinedLoss()
    loss, loss_dict = criterion(pred, target)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
