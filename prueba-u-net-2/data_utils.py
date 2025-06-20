import os
import cv2
import numpy as np
import json
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
import yaml

def create_directories(base_path):
    """Create necessary directory structure"""
    dirs = [
        'images/train', 'images/val', 'images/test',
        'masks/train', 'masks/val', 'masks/test',
        'results', 'checkpoints'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
    
    print(f"Created directory structure in {base_path}")

def yolo_polygon_to_mask(polygon_points, img_width, img_height):
    """
    Convert YOLO polygon format to binary mask
    
    Args:
        polygon_points: List of normalized coordinates [x1, y1, x2, y2, ...]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Binary mask as numpy array
    """
    # Convert normalized coordinates to pixel coordinates
    points = []
    for i in range(0, len(polygon_points), 2):
        x = int(polygon_points[i] * img_width)
        y = int(polygon_points[i + 1] * img_height)
        points.append([x, y])
    
    # Create empty mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Fill polygon
    if len(points) >= 3:
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    
    return mask

def process_yolo_labels(label_file, img_width, img_height):
    """
    Process YOLO label file and create combined binary mask
    
    Args:
        label_file: Path to YOLO label file
        img_width: Image width
        img_height: Image height
    
    Returns:
        Combined binary mask for all objects
    """
    combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if not os.path.exists(label_file):
        return combined_mask
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:  # class_id + at least 2 points (4 coordinates)
            continue
        
        class_id = int(parts[0])
        polygon_points = [float(x) for x in parts[1:]]
        
        # Create mask for this polygon
        polygon_mask = yolo_polygon_to_mask(polygon_points, img_width, img_height)
        
        # Add to combined mask
        combined_mask = np.maximum(combined_mask, polygon_mask)
    
    return combined_mask

def split_dataset(images_dir, labels_dir, output_base_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train/val/test sets
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO labels
        output_base_dir: Base directory for output
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    # Get all image files
    image_files = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(valid_extensions):
            image_files.append(filename)
    
    print(f"Found {len(image_files)} images")
    
    # Split the dataset
    train_files, temp_files = train_test_split(image_files, test_size=(val_ratio + test_ratio), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    print(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Create directories
    create_directories(output_base_dir)
    
    # Process each split
    for split_name, file_list in splits.items():
        print(f"Processing {split_name} set...")
        
        for filename in file_list:
            # Copy image
            src_image = os.path.join(images_dir, filename)
            dst_image = os.path.join(output_base_dir, 'images', split_name, filename)
            shutil.copy2(src_image, dst_image)
            
            # Process label and create mask
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_file = os.path.join(labels_dir, label_filename)
            
            # Get image dimensions
            img = Image.open(src_image)
            img_width, img_height = img.size
            
            # Create mask
            mask = process_yolo_labels(label_file, img_width, img_height)
            
            # Save mask
            mask_filename = os.path.splitext(filename)[0] + '_mask.png'
            mask_path = os.path.join(output_base_dir, 'masks', split_name, mask_filename)
            cv2.imwrite(mask_path, mask * 255)  # Save as 0-255 values
    
    # Save split information
    split_info = {
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
    }
    
    with open(os.path.join(output_base_dir, 'dataset_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return split_info

def resize_image_and_mask(image, mask, target_size=(1024, 1024)):
    """
    Resize image and mask to target size
    
    Args:
        image: PIL Image or numpy array
        mask: numpy array
        target_size: (width, height) tuple
    
    Returns:
        Resized image and mask
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize image
    image_resized = image.resize(target_size, Image.LANCZOS)
    
    # Resize mask
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(target_size, Image.NEAREST)
    mask_resized = np.array(mask_resized) / 255.0
    
    return image_resized, mask_resized 