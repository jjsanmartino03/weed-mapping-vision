#!/usr/bin/env python3
"""
Example usage script for the U-Net weed detection pipeline.

This script demonstrates how to:
1. Download and process data from Roboflow
2. Train a U-Net model
3. Test the trained model on new images

Make sure to replace 'YOUR_API_KEY_HERE' with your actual Roboflow API key.
"""

import os
import sys
import json

def run_training_example():
    """Example of running the complete training pipeline"""
    
    print("=" * 60)
    print("🌱 U-Net Weed Detection Training Example")
    print("=" * 60)
    
    # Configuration for the training
    config = {
        "workspace": "proyecto-finalmapeo-de-malezas",
        "project": "identificacion-malezas", 
        "version": 4,
        "raw_data_dir": "example_raw_dataset",
        "processed_data_dir": "example_processed_dataset",
        "output_dir": "example_training_output",
        "batch_size": 2,  # Smaller batch size for example
        "num_epochs": 10,  # Fewer epochs for quick example
        "learning_rate": 1e-4,
        "image_size": [512, 512],  # Smaller image size for faster training
        "num_workers": 2,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "test_ratio": 0.1
    }
    
    # Save example config
    config_path = "example_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Created example configuration: {config_path}")
    print("\n🚀 To run training, use:")
    print(f"python train_unet.py --api_key YOUR_API_KEY_HERE --config {config_path}")
    
    print("\n📋 Configuration details:")
    for key, value in config.items():
        print(f"  • {key}: {value}")

def run_testing_example():
    """Example of testing a trained model"""
    
    print("\n" + "=" * 60)
    print("🔍 Model Testing Example")
    print("=" * 60)
    
    # Example model testing commands
    model_path = "example_training_output/checkpoints/best_model.pth"
    test_image = "test_image.jpg"
    test_dir = "test_images/"
    
    print("📊 After training, you can test your model with:")
    print(f"\n1. Test single image:")
    print(f"   python test_model.py --model {model_path} --image {test_image}")
    
    print(f"\n2. Test directory of images:")
    print(f"   python test_model.py --model {model_path} --images_dir {test_dir}")
    
    print(f"\n3. Create overlay visualizations:")
    print(f"   python test_model.py --model {model_path} --image {test_image} --overlay")
    
    print(f"\n4. Adjust detection threshold:")
    print(f"   python test_model.py --model {model_path} --image {test_image} --threshold 0.3")

def show_pipeline_overview():
    """Show an overview of the complete pipeline"""
    
    print("\n" + "=" * 60)
    print("📋 Complete Pipeline Overview")
    print("=" * 60)
    
    steps = [
        ("1. Data Download", "Downloads dataset from Roboflow using API"),
        ("2. Data Processing", "Converts YOLO polygons to binary masks"),
        ("3. Dataset Splitting", "Splits data into train/val/test sets (70/20/10)"),
        ("4. Data Augmentation", "Applies geometric and color augmentations"),
        ("5. Model Training", "Trains U-Net with BCE + Dice loss"),
        ("6. Monitoring", "Logs metrics to TensorBoard and saves plots"),
        ("7. Model Selection", "Saves best model based on validation Dice score"),
        ("8. Evaluation", "Tests model on unseen images")
    ]
    
    for step, description in steps:
        print(f"{step:20} → {description}")

def show_file_structure():
    """Show the expected output file structure"""
    
    print("\n" + "=" * 60)
    print("📁 Output File Structure")
    print("=" * 60)
    
    structure = """
example_training_output/
├── checkpoints/
│   ├── best_model.pth              # Best model (highest val Dice)
│   └── checkpoint_epoch_*.pth      # Regular checkpoints
├── results/
│   ├── training_history.png        # Loss and metric plots
│   ├── training_metrics.json       # Numerical results
│   └── predictions_epoch_*.png     # Sample predictions
├── tensorboard/                    # TensorBoard logs
└── processed_dataset/              # Processed data
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── masks/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── dataset_split.json          # Split information
"""
    
    print(structure)

def show_requirements():
    """Show system requirements and setup"""
    
    print("\n" + "=" * 60)
    print("⚙️  System Requirements")
    print("=" * 60)
    
    requirements = [
        ("Python", "3.8+"),
        ("GPU Memory", "6GB+ recommended"),
        ("RAM", "8GB+ recommended"),
        ("Storage", "5GB+ for dataset and models"),
        ("CUDA", "For GPU acceleration (optional)")
    ]
    
    print("Minimum requirements:")
    for req, spec in requirements:
        print(f"  • {req:15} → {spec}")
    
    print(f"\n📦 Install dependencies:")
    print(f"  pip install -r requirements.txt")

def main():
    """Main example function"""
    
    print("🤖 U-Net Weed Detection Pipeline - Example Usage")
    print("=" * 60)
    
    # Show pipeline overview
    show_pipeline_overview()
    
    # Show system requirements
    show_requirements()
    
    # Show training example
    run_training_example()
    
    # Show testing example
    run_testing_example()
    
    # Show file structure
    show_file_structure()
    
    print("\n" + "=" * 60)
    print("✅ Ready to start! Follow the steps above to train your model.")
    print("💡 Tip: Start with smaller image sizes and fewer epochs for testing.")
    print("🔑 Don't forget to get your Roboflow API key!")
    print("=" * 60)

if __name__ == "__main__":
    main() 