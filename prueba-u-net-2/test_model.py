import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
from unet_model import UNet
from visualization import denormalize_image, create_overlay_visualization
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ModelTester:
    def __init__(self, model_path, config_path=None, device=None):
        """
        Initialize model tester
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to config file (to check if patches were used)
            device: Device to run inference on
        """
        # Load config if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Check if patches were used during training
        self.use_patches = self.config.get('patches', {}).get('enabled', False)
        self.patch_size = self.config.get('patches', {}).get('patch_size', 512)
        self.patch_overlap = self.config.get('patches', {}).get('overlap', 64)
        
        print(f"Patch-based inference: {'Enabled' if self.use_patches else 'Disabled'}")
        if self.use_patches:
            print(f"Patch size: {self.patch_size}x{self.patch_size}, Overlap: {self.patch_overlap}")
        
        # Proper device detection for Apple Silicon, CUDA, and CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Define preprocessing transform
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        model = UNet(n_channels=3, n_classes=1, bilinear=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Best validation Dice: {checkpoint.get('best_val_dice', 'N/A')}")
        
        return model
    
    def extract_patches(self, image, patch_size, overlap):
        """
        Extract overlapping patches from image
        
        Args:
            image: Input image (H, W, C)
            patch_size: Size of patches
            overlap: Overlap between patches
            
        Returns:
            patches: List of patches
            positions: List of (row, col) positions for reconstruction
        """
        h, w = image.shape[:2]
        stride = patch_size - overlap
        
        patches = []
        positions = []
        
        # Calculate patch positions to ensure full coverage
        row_positions = list(range(0, h - patch_size + 1, stride))
        col_positions = list(range(0, w - patch_size + 1, stride))
        
        # Add final positions to cover the entire image
        if len(row_positions) == 0 or row_positions[-1] + patch_size < h:
            row_positions.append(max(0, h - patch_size))
        if len(col_positions) == 0 or col_positions[-1] + patch_size < w:
            col_positions.append(max(0, w - patch_size))
        
        # Remove duplicates while preserving order
        row_positions = list(dict.fromkeys(row_positions))
        col_positions = list(dict.fromkeys(col_positions))
        
        for row in row_positions:
            for col in col_positions:
                # Ensure we don't go out of bounds
                end_row = min(row + patch_size, h)
                end_col = min(col + patch_size, w)
                start_row = max(end_row - patch_size, 0)
                start_col = max(end_col - patch_size, 0)
                
                patch = image[start_row:end_row, start_col:end_col]
                patches.append(patch)
                positions.append((start_row, start_col))
        
        return patches, positions
    
    def reconstruct_from_patches(self, patch_predictions, positions, original_shape, patch_size, overlap):
        """
        Reconstruct full prediction from patch predictions
        
        Args:
            patch_predictions: List of patch predictions
            positions: List of patch positions
            original_shape: Shape of original image (H, W)
            patch_size: Size of patches
            overlap: Overlap between patches
            
        Returns:
            Reconstructed prediction
        """
        h, w = original_shape[:2]
        prediction = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        for pred, (row, col) in zip(patch_predictions, positions):
            end_row = min(row + patch_size, h)
            end_col = min(col + patch_size, w)
            
            # Create weight map for blending (higher weight in center)
            patch_weight = np.ones_like(pred, dtype=np.float32)
            if overlap > 0:
                # Reduce weight at edges for smooth blending
                fade_size = overlap // 2
                for i in range(fade_size):
                    weight = (i + 1) / fade_size
                    patch_weight[i, :] *= weight  # Top edge
                    patch_weight[-i-1, :] *= weight  # Bottom edge
                    patch_weight[:, i] *= weight  # Left edge
                    patch_weight[:, -i-1] *= weight  # Right edge
            
            prediction[row:end_row, col:end_col] += pred * patch_weight
            weight_map[row:end_row, col:end_col] += patch_weight
        
        # Normalize by weight map
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        prediction = prediction / weight_map
        
        return prediction

    def preprocess_image(self, image_path, target_size=None):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to input image
            target_size: Target size for resizing (only used if not using patches)
        
        Returns:
            Preprocessed image tensor and original image
        """
        # Load image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        if self.use_patches:
            # For patch-based inference, keep original resolution
            return original_image, original_image
        else:
            # For full-image inference, resize
            if target_size is None:
                target_size = (1024, 1024)
            image = cv2.resize(original_image, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Apply transforms
            transformed = self.transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
            
            return image_tensor, original_image

    def predict_patch(self, patch):
        """
        Predict on a single patch
        
        Args:
            patch: Input patch (H, W, C)
            
        Returns:
            Prediction probabilities
        """
        # Apply transforms
        transformed = self.transform(image=patch)
        patch_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(patch_tensor)
            probabilities = torch.sigmoid(logits)
        
        return probabilities.cpu().squeeze().numpy()

    def predict(self, image_path, threshold=0.5):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binary classification
        
        Returns:
            Prediction mask, confidence map, and processed image
        """
        if self.use_patches:
            return self.predict_with_patches(image_path, threshold)
        else:
            return self.predict_full_image(image_path, threshold)
    
    def predict_with_patches(self, image_path, threshold=0.5):
        """
        Make prediction using patch-based approach
        """
        # Load original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        print(f"Processing image: {image_path}")
        print(f"Original size: {original_image.shape[:2]}")
        
        # Extract patches
        patches, positions = self.extract_patches(
            original_image, self.patch_size, self.patch_overlap
        )
        
        print(f"Extracted {len(patches)} patches")
        
        # Predict on each patch
        patch_predictions = []
        for i, patch in enumerate(patches):
            if i % 50 == 0:  # Progress update
                print(f"Processing patch {i+1}/{len(patches)}")
            
            pred_prob = self.predict_patch(patch)
            patch_predictions.append(pred_prob)
            
            # Clear GPU cache periodically
            if self.device.type == 'mps' and i % 10 == 0:
                torch.mps.empty_cache()
        
        # Reconstruct full prediction
        probabilities = self.reconstruct_from_patches(
            patch_predictions, positions, original_image.shape, 
            self.patch_size, self.patch_overlap
        )
        
        # Apply threshold
        prediction = (probabilities > threshold).astype(np.float32)
        
        print("Patch-based prediction completed!")
        
        return prediction, probabilities, original_image, original_image
    
    def predict_full_image(self, image_path, threshold=0.5):
        """
        Make prediction on full resized image (original method)
        """
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits)
            prediction = (probabilities > threshold).float()
        
        # Move to CPU and convert to numpy
        probabilities = probabilities.cpu().squeeze().numpy()
        prediction = prediction.cpu().squeeze().numpy()
        
        # Get resized image for visualization
        resized_image = cv2.resize(original_image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        
        return prediction, probabilities, resized_image, original_image
    
    def predict_batch(self, image_paths, threshold=0.5):
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of image paths
            threshold: Threshold for binary classification
        
        Returns:
            List of predictions
        """
        results = []
        
        for image_path in image_paths:
            prediction, probabilities, resized_image, original_image = self.predict(image_path, threshold)
            results.append({
                'image_path': image_path,
                'prediction': prediction,
                'probabilities': probabilities,
                'resized_image': resized_image,
                'original_image': original_image
            })
        
        return results
    
    def visualize_prediction(self, image_path, threshold=0.5, save_path=None):
        """
        Visualize prediction for a single image
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binary classification
            save_path: Path to save visualization
        """
        # Make prediction
        prediction, probabilities, resized_image, original_image = self.predict(image_path, threshold)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(resized_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Confidence map
        im1 = axes[1].imshow(probabilities, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Confidence Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Binary prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title(f'Prediction (threshold={threshold})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def create_overlay(self, image_path, threshold=0.5, save_path=None):
        """
        Create overlay visualization
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binary classification
            save_path: Path to save overlay
        """
        # Make prediction
        prediction, probabilities, resized_image, original_image = self.predict(image_path, threshold)
        
        # Create overlay (red for predicted weeds)
        overlay = resized_image.copy()
        overlay[prediction > threshold] = [255, 0, 0]  # Red overlay for weeds
        
        # Blend with original image
        alpha = 0.3
        result = cv2.addWeighted(resized_image, 1-alpha, overlay, alpha, 0)
        
        # Display
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(resized_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title('Weed Detection Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overlay saved to: {save_path}")
            
            # Also save just the overlay image
            overlay_path = save_path.replace('.png', '_overlay_only.png')
            cv2.imwrite(overlay_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        plt.show()
        
        # Print some statistics
        total_pixels = prediction.shape[0] * prediction.shape[1]
        weed_pixels = np.sum(prediction > threshold)
        weed_percentage = (weed_pixels / total_pixels) * 100
        
        print(f"Weed coverage: {weed_percentage:.2f}% ({weed_pixels}/{total_pixels} pixels)")

def main():
    parser = argparse.ArgumentParser(description='Test trained U-Net model on images')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file (to check if patches were used)')
    parser.add_argument('--image', type=str, help='Path to single image for testing')
    parser.add_argument('--images_dir', type=str, help='Directory containing images for batch testing')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    parser.add_argument('--overlay', action='store_true', help='Create overlay visualization')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = ModelTester(args.model, args.config)
    
    if args.image:
        # Test single image
        print(f"Testing single image: {args.image}")
        
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        
        if args.overlay:
            save_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
            tester.create_overlay(args.image, args.threshold, save_path)
        else:
            save_path = os.path.join(args.output_dir, f"{base_name}_prediction.png")
            tester.visualize_prediction(args.image, args.threshold, save_path)
    
    elif args.images_dir:
        # Test directory of images
        print(f"Testing images in directory: {args.images_dir}")
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = []
        
        for filename in os.listdir(args.images_dir):
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(args.images_dir, filename))
        
        print(f"Found {len(image_paths)} images")
        
        # Process each image
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            if args.overlay:
                save_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
                tester.create_overlay(image_path, args.threshold, save_path)
            else:
                save_path = os.path.join(args.output_dir, f"{base_name}_prediction.png")
                tester.visualize_prediction(image_path, args.threshold, save_path)
    
    else:
        print("Please provide either --image or --images_dir argument")

if __name__ == "__main__":
    main() 