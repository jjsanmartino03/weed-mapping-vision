import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WeedSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(1024, 1024), transform=None):
        """
        Dataset class for weed segmentation
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            image_size: Target size for images (width, height)
            transform: Albumentations transform pipeline
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(valid_extensions):
                self.image_files.append(filename)
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = os.path.splitext(img_name)[0] + '_mask.png'
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask / 255.0).astype(np.float32)  # Normalize to 0-1 and ensure float32
        else:
            # Create empty mask if no mask file exists
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Resize to target size
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensor if not already done by transform (ensure float32 for MPS compatibility)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask.astype(np.float32)).float()
        
        # Ensure float32 dtype for MPS compatibility
        image = image.float()
        mask = mask.float()
        
        # Add channel dimension to mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask

def get_transforms(image_size, is_training=True):
    """
    Get augmentation transforms for training or validation
    
    Args:
        image_size: Target image size (width, height)
        is_training: Whether to apply training augmentations
    
    Returns:
        Albumentations transform pipeline
    """
    if is_training:
        transform = A.Compose([
            A.Resize(image_size[1], image_size[0]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.3
            ),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size[1], image_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform

def create_data_loaders(data_dir, batch_size=4, image_size=(1024, 1024), num_workers=4):
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        data_dir: Base directory containing images and masks folders
        batch_size: Batch size for data loaders
        image_size: Target image size (width, height)
        num_workers: Number of worker processes for data loading
    
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    datasets = {}
    data_loaders = {}
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = os.path.join(data_dir, 'images', split)
        masks_dir = os.path.join(data_dir, 'masks', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"Warning: {split} directories not found, skipping...")
            continue
        
        is_training = (split == 'train')
        transform = get_transforms(image_size, is_training)
        
        dataset = WeedSegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            image_size=image_size,
            transform=transform
        )
        
        datasets[split] = dataset
        
        # Create data loader (disable pin_memory for MPS compatibility)
        use_pin_memory = torch.cuda.is_available()  # Only use pin_memory with CUDA
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            drop_last=is_training
        )
        
        data_loaders[split] = data_loader
        
        print(f"{split.capitalize()} dataset: {len(dataset)} samples")
    
    return data_loaders, datasets 