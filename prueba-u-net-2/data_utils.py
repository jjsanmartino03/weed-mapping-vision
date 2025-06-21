import os
import cv2
import numpy as np
import json
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
from collections import defaultdict
from patch_creator import PatchDatasetCreator

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

def create_patch_dataset(config, processed_data_dir):
    """
    Create patch dataset from processed images and masks
    
    Args:
        config: Configuration dictionary with patch settings
        processed_data_dir: Directory containing processed images and masks
    
    Returns:
        Path to created patch dataset
    """
    patch_config = config['patches']
    
    if not patch_config['enabled']:
        print("Patches disabled in config. Using original dataset.")
        return processed_data_dir
    
    print("🔧 Creating patch dataset...")
    
    # Create patch dataset creator
    creator = PatchDatasetCreator(
        patch_size=patch_config['patch_size'],
        overlap=patch_config['overlap'],
        min_weed_pixels=patch_config['min_weed_pixels']
    )
    
    # Create patch dataset for each split
    patch_base_dir = os.path.join(os.path.dirname(processed_data_dir), 'patch_dataset')
    os.makedirs(patch_base_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    all_patch_stats = {}
    
    for split in splits:
        print(f"\n📁 Processing {split} split...")
        
        images_dir = os.path.join(processed_data_dir, 'images', split)
        masks_dir = os.path.join(processed_data_dir, 'masks', split)
        output_dir = os.path.join(patch_base_dir, split)
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"Warning: {split} split not found, skipping")
            continue
        
        # Create patches for this split
        patches = creator.create_patch_dataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            output_dir=output_dir,
            balance_strategy=patch_config['balance_strategy']
        )
        
        # Save patch statistics
        all_patch_stats[split] = {
            'total_patches': len(patches),
            'weed_patches': sum(1 for p in patches if p['has_weeds']),
            'background_patches': sum(1 for p in patches if not p['has_weeds'])
        }
    
    # Create proper directory structure for compatibility with existing data loader
    patch_output_dir = patch_base_dir
    
    # Create directories for each split
    for split in splits:
        os.makedirs(os.path.join(patch_output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(patch_output_dir, 'masks', split), exist_ok=True)
    
    # Move patches to proper structure
    patch_count = 0
    unified_split_info = {'train': [], 'val': [], 'test': []}
    
    for split in splits:
        split_source_dir = os.path.join(patch_base_dir, split)
        if not os.path.exists(split_source_dir):
            continue
            
        split_images_source = os.path.join(split_source_dir, 'images')
        split_masks_source = os.path.join(split_source_dir, 'masks')
        
        split_images_dest = os.path.join(patch_output_dir, 'images', split)
        split_masks_dest = os.path.join(patch_output_dir, 'masks', split)
        
        if os.path.exists(split_images_source):
            for img_file in os.listdir(split_images_source):
                if img_file.endswith('.png'):
                    # Move image
                    src_img = os.path.join(split_images_source, img_file)
                    dst_img = os.path.join(split_images_dest, img_file)
                    shutil.move(src_img, dst_img)
                    
                    # Move corresponding mask
                    mask_file = img_file.replace('.png', '_mask.png')
                    src_mask = os.path.join(split_masks_source, mask_file)
                    dst_mask = os.path.join(split_masks_dest, mask_file)
                    
                    if os.path.exists(src_mask):
                        shutil.move(src_mask, dst_mask)
                        unified_split_info[split].append(img_file)
                        patch_count += 1
        
        # Remove empty split directories
        if os.path.exists(split_source_dir):
            shutil.rmtree(split_source_dir)
    
    # Save patch dataset info
    patch_info = {
        'patch_config': patch_config,
        'statistics': all_patch_stats,
        'total_patches': patch_count,
        'splits': unified_split_info
    }
    
    with open(os.path.join(patch_output_dir, 'patch_info.json'), 'w') as f:
        json.dump(patch_info, f, indent=2)
    
    print(f"\n✅ Patch dataset created successfully!")
    print(f"📊 Total patches: {patch_count}")
    print(f"📁 Location: {patch_output_dir}")
    
    # Print statistics summary
    total_weed = sum(stats['weed_patches'] for stats in all_patch_stats.values())
    total_bg = sum(stats['background_patches'] for stats in all_patch_stats.values())
    
    print(f"🌿 Weed patches: {total_weed}")
    print(f"🌱 Background patches: {total_bg}")
    print(f"⚖️ Balance ratio: {total_weed/(total_weed+total_bg)*100:.1f}% weeds")
    
    return patch_output_dir

def prepare_dataset(config):
    """
    Prepare dataset for training - handles both regular and patch datasets
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to prepared dataset
    """
    processed_data_dir = config['processed_data_dir']
    
    # Check if patches are enabled
    if config.get('patches', {}).get('enabled', False):
        print("🔧 Patch mode enabled - creating patch dataset...")
        return create_patch_dataset(config, processed_data_dir)
    else:
        print("📁 Using regular dataset (no patches)")
        return processed_data_dir 