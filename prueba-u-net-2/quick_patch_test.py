from patch_creator import PatchDatasetCreator
import cv2
import matplotlib.pyplot as plt
import numpy as np

def compare_approaches():
    """Quick visual comparison of original vs patch approach"""
    
    # Load a sample image
    img_path = "processed_dataset/images/train/DJI_20241107174514_0388_D_JPG.rf.68bf96305540b97b2ab97ba0af05b175.jpg"
    mask_path = "processed_dataset/masks/train/DJI_20241107174514_0388_D_JPG.rf.68bf96305540b97b2ab97ba0af05b175_mask.png"
    
    # Load full resolution
    img_full = cv2.imread(img_path)
    img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
    mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Original image size: {img_full.shape}")
    print(f"Original weed pixels: {np.sum(mask_full > 127)}")
    
    # Create patches
    creator = PatchDatasetCreator(patch_size=512, overlap=64, min_weed_pixels=50)
    patches = creator.create_patches_from_image(img_path, mask_path)
    
    weed_patches = [p for p in patches if p['has_weeds']]
    bg_patches = [p for p in patches if not p['has_weeds']]
    
    print(f"\nPatches created: {len(patches)}")
    print(f"Weed patches: {len(weed_patches)}")
    print(f"Background patches: {len(bg_patches)}")
    
    # Show comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original downsampled
    img_small = cv2.resize(img_full, (512, 512))
    mask_small = cv2.resize(mask_full, (512, 512))
    
    axes[0,0].imshow(img_small)
    axes[0,0].set_title("Original (downsampled to 512x512)")
    axes[0,0].axis('off')
    
    axes[0,1].imshow(mask_small, cmap='gray')
    axes[0,1].set_title("Mask (downsampled)")
    axes[0,1].axis('off')
    
    # Show patch examples
    if len(weed_patches) >= 2:
        # Weed patch 1
        axes[1,0].imshow(weed_patches[0]['image_patch'])
        axes[1,0].set_title(f"Patch 1 (Weeds: {weed_patches[0]['weed_pixels']}px)")
        axes[1,0].axis('off')
        
        axes[1,1].imshow(weed_patches[0]['mask_patch'], cmap='gray')
        axes[1,1].set_title("Patch 1 Mask")
        axes[1,1].axis('off')
        
        # Weed patch 2
        if len(weed_patches) > 1:
            axes[1,2].imshow(weed_patches[1]['image_patch'])
            axes[1,2].set_title(f"Patch 2 (Weeds: {weed_patches[1]['weed_pixels']}px)")
            axes[1,2].axis('off')
        else:
            axes[1,2].axis('off')
    
    # Background patch
    if len(bg_patches) > 0:
        axes[0,2].imshow(bg_patches[0]['image_patch'])
        axes[0,2].set_title("Background Patch")
        axes[0,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('patch_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison saved as 'patch_comparison.png'")
    
    # Calculate improvement potential
    print(f"\n🎯 IMPROVEMENT POTENTIAL:")
    print(f"Current approach: {np.sum(mask_small > 127)} weed pixels in 512x512")
    print(f"Patch approach: {sum(p['weed_pixels'] for p in weed_patches)} weed pixels total")
    print(f"Detail improvement: {sum(p['weed_pixels'] for p in weed_patches) / max(1, np.sum(mask_small > 127)):.1f}x more weed detail")

if __name__ == "__main__":
    compare_approaches() 