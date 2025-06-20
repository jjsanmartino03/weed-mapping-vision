import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from unet_model import UNet
from visualization import denormalize_image, create_overlay_visualization
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ModelTester:
    def __init__(self, model_path, device=None):
        """
        Initialize model tester
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
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
    
    def preprocess_image(self, image_path, target_size=(1024, 1024)):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to input image
            target_size: Target size for resizing
        
        Returns:
            Preprocessed image tensor and original image
        """
        # Load image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(original_image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image, image
    
    def predict(self, image_path, threshold=0.5):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binary classification
        
        Returns:
            Prediction mask, confidence map, and processed image
        """
        # Preprocess image
        image_tensor, original_image, resized_image = self.preprocess_image(image_path)
        
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
        overlay[prediction > 0.5] = [255, 0, 0]  # Red overlay for weeds
        
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
        weed_pixels = np.sum(prediction > 0.5)
        weed_percentage = (weed_pixels / total_pixels) * 100
        
        print(f"Weed coverage: {weed_percentage:.2f}% ({weed_pixels}/{total_pixels} pixels)")

def main():
    parser = argparse.ArgumentParser(description='Test trained U-Net model on images')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image for testing')
    parser.add_argument('--images_dir', type=str, help='Directory containing images for batch testing')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    parser.add_argument('--overlay', action='store_true', help='Create overlay visualization')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = ModelTester(args.model)
    
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