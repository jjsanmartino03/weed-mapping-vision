import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply sigmoid to predictions
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice_coeff

class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth)
    
    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    """IoU Loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply sigmoid to predictions
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return IoU loss (1 - IoU)
        return 1 - iou

def dice_coefficient(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice coefficient for evaluation
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth masks
        threshold: Threshold for converting predictions to binary
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient
    """
    # Apply sigmoid and threshold
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    # Flatten tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate Dice coefficient
    intersection = (predictions * targets).sum()
    dice_coeff = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return dice_coeff

def iou_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate IoU score for evaluation
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth masks
        threshold: Threshold for converting predictions to binary
        smooth: Smoothing factor
    
    Returns:
        IoU score
    """
    # Apply sigmoid and threshold
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    # Flatten tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection and union
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou 