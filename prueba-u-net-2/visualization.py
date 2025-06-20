import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def plot_training_history(train_losses, val_losses, train_dices, val_dices, train_ious, val_ious, save_path=None):
    """
    Plot training history including losses and metrics
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_dices: List of training Dice scores
        val_dices: List of validation Dice scores
        train_ious: List of training IoU scores
        val_ious: List of validation IoU scores
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Dice scores
    axes[1].plot(epochs, train_dices, 'b-', label='Training Dice', linewidth=2)
    axes[1].plot(epochs, val_dices, 'r-', label='Validation Dice', linewidth=2)
    axes[1].set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot IoU scores
    axes[2].plot(epochs, train_ious, 'b-', label='Training IoU', linewidth=2)
    axes[2].plot(epochs, val_ious, 'r-', label='Validation IoU', linewidth=2)
    axes[2].set_title('Training and Validation IoU Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
    
    Returns:
        Denormalized image tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor

def save_sample_predictions(model, data_loader, device, save_path, num_samples=4):
    """
    Save sample predictions for visualization
    
    Args:
        model: Trained model
        data_loader: Data loader for getting samples
        device: Device to run inference on
        save_path: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get a batch of data
    data_iter = iter(data_loader)
    images, masks = next(data_iter)
    
    # Select samples (limit to available batch size)
    available_samples = min(num_samples, images.size(0))
    indices = torch.randperm(images.size(0))[:available_samples]
    sample_images = images[indices]
    sample_masks = masks[indices]
    
    # Move to device and get predictions
    sample_images = sample_images.to(device)
    sample_masks = sample_masks.to(device)
    
    with torch.no_grad():
        predictions = model(sample_images)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
    
    # Move back to CPU for visualization
    sample_images = sample_images.cpu()
    sample_masks = sample_masks.cpu()
    predictions = predictions.cpu()
    
    # Create visualization with actual number of samples
    num_samples = available_samples
    if num_samples == 1:
        # Special case for single sample - create vertical layout
        fig, axes = plt.subplots(3, 1, figsize=(6, 12))
        axes = axes.reshape(3, 1)  # Ensure 2D array for consistent indexing
    else:
        fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    
    for i in range(num_samples):
        # Denormalize image
        img = denormalize_image(sample_images[i])
        img = img.permute(1, 2, 0).numpy()
        
        # Get mask and prediction
        mask = sample_masks[i, 0].numpy()
        pred = predictions[i, 0].numpy()
        
        # Plot original image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[0, i].axis('off')
        
        # Plot ground truth mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Ground Truth {i+1}')
        axes[1, i].axis('off')
        
        # Plot prediction
        axes[2, i].imshow(pred, cmap='gray')
        axes[2, i].set_title(f'Prediction {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sample predictions saved to: {save_path}")

def visualize_dataset_samples(dataset, num_samples=8, save_path=None):
    """
    Visualize samples from the dataset
    
    Args:
        dataset: Dataset object
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 6))
    
    for i in range(num_samples):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        image, mask = dataset[idx]
        
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # CHW format
                image = image.permute(1, 2, 0)
            image = image.numpy()
        
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().numpy()
        
        # Plot image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Plot mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset samples visualization saved to: {save_path}")
    
    plt.show()

def create_overlay_visualization(image, mask, prediction, alpha=0.5):
    """
    Create overlay visualization of image with mask and prediction
    
    Args:
        image: Original image (numpy array)
        mask: Ground truth mask (numpy array)
        prediction: Model prediction (numpy array)
        alpha: Transparency for overlay
    
    Returns:
        Overlay visualization
    """
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create colored overlays
    mask_overlay = np.zeros_like(image)
    pred_overlay = np.zeros_like(image)
    
    # Green for ground truth, Red for prediction
    mask_overlay[mask > 0.5] = [0, 255, 0]  # Green
    pred_overlay[prediction > 0.5] = [255, 0, 0]  # Red
    
    # Create final overlay
    overlay = image.copy()
    overlay = cv2.addWeighted(overlay, 1-alpha, mask_overlay, alpha, 0)
    overlay = cv2.addWeighted(overlay, 1-alpha, pred_overlay, alpha, 0)
    
    return overlay 