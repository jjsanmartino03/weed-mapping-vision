import os
import cv2
import numpy as np
from patchify import patchify
import json
from tqdm import tqdm
from collections import defaultdict

class PatchDatasetCreator:
    def __init__(self, patch_size=512, overlap=64, min_weed_pixels=50):
        """
        Create patches from high-resolution images
        
        Args:
            patch_size: Size of square patches (512, 768, etc.)
            overlap: Overlap between patches in pixels
            min_weed_pixels: Minimum weed pixels to keep a patch
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.min_weed_pixels = min_weed_pixels
        
    def create_patches_from_image(self, image_path, mask_path):
        """Extract patches from a single image and its mask"""
        # Load image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        h, w = image.shape[:2]
        
        # Calculate how many patches fit
        h_patches = (h - self.overlap) // self.stride
        w_patches = (w - self.overlap) // self.stride
        
        patches_data = []
        
        for i in range(h_patches):
            for j in range(w_patches):
                # Calculate patch coordinates
                y1 = i * self.stride
                x1 = j * self.stride
                y2 = min(y1 + self.patch_size, h)
                x2 = min(x1 + self.patch_size, w)
                
                # Skip if patch is too small
                if (y2 - y1) < self.patch_size or (x2 - x1) < self.patch_size:
                    continue
                
                # Extract patches
                img_patch = image[y1:y2, x1:x2]
                mask_patch = mask[y1:y2, x1:x2]
                
                # Calculate weed statistics
                weed_pixels = np.sum(mask_patch > 0.5)
                total_pixels = self.patch_size * self.patch_size
                weed_ratio = weed_pixels / total_pixels
                
                patches_data.append({
                    'image_patch': img_patch,
                    'mask_patch': mask_patch,
                    'weed_pixels': int(weed_pixels),
                    'weed_ratio': float(weed_ratio),
                    'coordinates': (y1, x1, y2, x2),
                    'has_weeds': weed_pixels >= self.min_weed_pixels
                })
        
        return patches_data
    
    def balance_patches(self, all_patches, strategy='weed_focused'):
        """
        Balance patches between weed and no-weed patches
        
        Strategies:
        - 'balanced': 50% weed patches, 50% background
        - 'weed_focused': 70% weed patches, 30% background  
        - 'all_weed': Only patches with weeds
        - 'natural': Keep all patches (natural distribution)
        """
        weed_patches = [p for p in all_patches if p['has_weeds']]
        background_patches = [p for p in all_patches if not p['has_weeds']]
        
        print(f"Found {len(weed_patches)} weed patches, {len(background_patches)} background patches")
        
        if strategy == 'balanced':
            # Equal number of weed and background patches
            n_weed = len(weed_patches)
            if len(background_patches) >= n_weed:
                selected_bg = np.random.choice(len(background_patches), n_weed, replace=False)
                selected_background = [background_patches[i] for i in selected_bg]
            else:
                selected_background = background_patches
            return weed_patches + selected_background
            
        elif strategy == 'weed_focused':
            # 70% weed, 30% background
            n_weed = len(weed_patches)
            n_background = min(int(n_weed * 0.43), len(background_patches))  # 30/70 = 0.43
            if n_background > 0:
                selected_bg = np.random.choice(len(background_patches), n_background, replace=False)
                selected_background = [background_patches[i] for i in selected_bg]
            else:
                selected_background = []
            return weed_patches + selected_background
            
        elif strategy == 'all_weed':
            # Only weed patches
            return weed_patches
            
        else:  # 'natural'
            # Keep all patches
            return all_patches
    
    def create_patch_dataset(self, images_dir, masks_dir, output_dir, 
                           balance_strategy='weed_focused'):
        """Create complete patch dataset"""
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        all_patches = []
        patch_stats = defaultdict(int)
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(images_dir, img_file)
            
            # Find corresponding mask
            mask_file = img_file.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
            mask_path = os.path.join(masks_dir, mask_file)
            
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {img_file}")
                continue
            
            # Extract patches from this image
            patches = self.create_patches_from_image(img_path, mask_path)
            
            # Add source image info
            for patch in patches:
                patch['source_image'] = img_file
            
            all_patches.extend(patches)
            
            # Update statistics
            patch_stats['total_patches'] += len(patches)
            patch_stats['weed_patches'] += sum(1 for p in patches if p['has_weeds'])
            patch_stats['background_patches'] += sum(1 for p in patches if not p['has_weeds'])
        
        print(f"\nTotal patches extracted: {len(all_patches)}")
        print(f"Weed patches: {patch_stats['weed_patches']}")
        print(f"Background patches: {patch_stats['background_patches']}")
        
        # Balance the dataset
        balanced_patches = self.balance_patches(all_patches, balance_strategy)
        
        print(f"\nAfter balancing ({balance_strategy}): {len(balanced_patches)} patches")
        
        # Save patches
        for idx, patch_data in enumerate(tqdm(balanced_patches, desc="Saving patches")):
            # Save image patch
            img_filename = f"patch_{idx:06d}.png"
            img_path = os.path.join(output_dir, 'images', img_filename)
            cv2.imwrite(img_path, cv2.cvtColor(patch_data['image_patch'], cv2.COLOR_RGB2BGR))
            
            # Save mask patch
            mask_filename = f"patch_{idx:06d}_mask.png"
            mask_path = os.path.join(output_dir, 'masks', mask_filename)
            cv2.imwrite(mask_path, (patch_data['mask_patch'] * 255).astype(np.uint8))
        
        # Save dataset info
        dataset_info = {
            'patch_size': self.patch_size,
            'overlap': self.overlap,
            'stride': self.stride,
            'min_weed_pixels': self.min_weed_pixels,
            'balance_strategy': balance_strategy,
            'total_patches': len(balanced_patches),
            'original_images': len(image_files),
            'statistics': dict(patch_stats)
        }
        
        with open(os.path.join(output_dir, 'patch_dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✅ Patch dataset created successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Total patches: {len(balanced_patches)}")
        
        return balanced_patches

if __name__ == "__main__":
    # Example usage
    creator = PatchDatasetCreator(
        patch_size=512,
        overlap=64,
        min_weed_pixels=100
    )
    
    # Update these paths to match your dataset
    images_dir = "processed_dataset/images/train"
    masks_dir = "processed_dataset/masks/train" 
    output_dir = "patch_dataset"
    
    creator.create_patch_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        balance_strategy='weed_focused'  # 70% weed, 30% background
    ) 