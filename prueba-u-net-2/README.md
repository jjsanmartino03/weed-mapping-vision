# U-Net Training Pipeline for Weed Detection in UAV Imagery

This repository contains a complete U-Net training pipeline for semantic segmentation of weeds in UAV agricultural imagery. The pipeline handles data download from Roboflow, YOLO polygon to mask conversion, dataset splitting, and U-Net model training.

## 🚀 Features

- **Automated Data Pipeline**: Downloads and processes data from Roboflow
- **YOLO to Mask Conversion**: Converts YOLO polygon annotations to binary masks
- **Patch-Based Training**: Optional patch extraction for high-resolution images
- **Data Augmentation**: Comprehensive augmentation pipeline using Albumentations
- **Advanced Loss Functions**: BCE + Dice Loss combination for better segmentation
- **Training Monitoring**: TensorBoard logging and visualization utilities
- **Model Checkpointing**: Automatic saving of best models based on validation metrics
- **Intelligent Testing**: Patch-based inference for models trained with patches

## 📁 Project Structure

```
prueba-u-net-2/
├── train_unet.py              # Main training script
├── unet_model.py              # U-Net model definition
├── dataset.py                 # PyTorch dataset and data loaders
├── data_utils.py              # Data processing utilities
├── patch_creator.py           # Patch extraction utilities
├── losses.py                  # Loss functions and metrics
├── visualization.py           # Visualization utilities
├── test_model.py              # Model testing and inference
├── example_patch_testing.py   # Example patch-based testing
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Installation

1. **Clone the repository and navigate to the directory:**
   ```bash
   cd prueba-u-net-2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Basic Usage

Run the training pipeline with your Roboflow API key:

```bash
python train_unet.py --api_key YOUR_ROBOFLOW_API_KEY
```

### Patch-Based Training

For high-resolution images, enable patch-based training in your config:

```json
{
  "patches": {
    "enabled": true,
    "patch_size": 512,
    "overlap": 64,
    "balance_strategy": "weed_focused",
    "min_weed_pixels": 10
  }
}
```

### Model Testing

Test your trained model on new images:

```bash
# Test single image with automatic patch detection
python test_model.py --model training_output/best_model.pth --config config.json --image test_image.jpg --overlay

# Test directory of images
python test_model.py --model training_output/best_model.pth --config config.json --images_dir test_images/ --overlay
```

### Example Testing Script

Use the provided example for patch-based testing:

```bash
python example_patch_testing.py
```

### Advanced Usage

You can customize the training by modifying the `config.json` file or passing arguments:

```bash
python train_unet.py --api_key YOUR_API_KEY --config custom_config.json
```

### Configuration Options

The `config.json` file contains all training parameters:

- **Dataset Settings:**
  - `workspace`: Roboflow workspace name
  - `project`: Roboflow project name
  - `version`: Dataset version number
  - `train_ratio`: Training set proportion (default: 0.7)
  - `val_ratio`: Validation set proportion (default: 0.2)
  - `test_ratio`: Test set proportion (default: 0.1)

- **Training Settings:**
  - `batch_size`: Batch size for training (default: 4)
  - `num_epochs`: Number of training epochs (default: 50)
  - `learning_rate`: Initial learning rate (default: 1e-4)
  - `image_size`: Input image size [width, height] (default: [1024, 1024])

- **Model Settings:**
  - `bce_weight`: Weight for BCE loss (default: 0.5)
  - `dice_weight`: Weight for Dice loss (default: 0.5)
  - `bilinear`: Use bilinear upsampling (default: false)

## 📊 What the Pipeline Does

### 1. Data Download and Organization
- Downloads dataset from Roboflow in YOLOv8 segmentation format
- Automatically splits data into train/val/test sets
- Creates organized directory structure

### 2. YOLO to Mask Conversion
- Converts YOLO polygon annotations to binary masks
- Handles multiple polygons per image
- Saves masks as PNG files for efficient loading

### 3. Data Loading and Augmentation
- Implements PyTorch Dataset class for efficient data loading
- Applies comprehensive augmentations:
  - Geometric: Horizontal/vertical flips, rotations, scaling
  - Color: Brightness, contrast, saturation adjustments
  - Noise: Gaussian blur

### 4. Model Training
- U-Net architecture optimized for binary segmentation
- Combined BCE + Dice Loss for better performance
- Adam optimizer with learning rate scheduling
- Early stopping based on validation metrics

### 5. Monitoring and Visualization
- TensorBoard logging for training metrics
- Automatic saving of sample predictions
- Training history plots (loss, IoU, Dice score)
- Model checkpointing with best model selection

## 📈 Output Structure

After training, the following structure is created:

```
training_output/
├── checkpoints/
│   ├── best_model.pth           # Best model based on validation Dice
│   └── checkpoint_epoch_*.pth   # Regular checkpoints
├── results/
│   ├── training_history.png     # Training curves
│   ├── training_metrics.json    # Numerical results
│   └── predictions_epoch_*.png  # Sample predictions
├── tensorboard/                 # TensorBoard logs
└── processed_dataset/           # Processed data
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/
        ├── val/
        └── test/
```

## 🔧 Model Architecture

The U-Net model includes:
- **Encoder**: 4 downsampling blocks with skip connections
- **Decoder**: 4 upsampling blocks with concatenated features
- **Output**: Single channel for binary segmentation
- **Features**: Batch normalization, ReLU activation, dropout

## 📊 Metrics and Loss Functions

- **Loss Function**: Combined BCE + Dice Loss
- **Metrics**: IoU (Intersection over Union), Dice Coefficient
- **Evaluation**: Threshold-based binary classification metrics

## 🎛️ Hyperparameters

Key hyperparameters and their defaults:
- Input size: 1024×1024 pixels
- Batch size: 4 (adjust based on GPU memory)
- Learning rate: 1e-4 with ReduceLROnPlateau scheduling
- Weight decay: 1e-4
- Loss weights: BCE=0.5, Dice=0.5

## 🚨 Requirements

- **Hardware**: GPU recommended (CUDA compatible)
- **Memory**: At least 8GB RAM, 6GB GPU memory for default settings
- **Python**: 3.8+
- **Key Dependencies**: PyTorch, OpenCV, Albumentations, Roboflow

## 🔍 Troubleshooting

**Common Issues:**

1. **Out of Memory**: Reduce batch_size in config.json
2. **Slow Training**: Reduce image_size or use more num_workers
3. **Poor Performance**: Increase num_epochs or adjust loss weights
4. **Data Download Issues**: Check your Roboflow API key and internet connection

## 📝 Example Results

The pipeline automatically generates:
- Training curves showing loss and metric progression
- Sample predictions comparing ground truth vs model output
- Comprehensive metrics logging via TensorBoard
- Best model selection based on validation performance

## 🤝 Contributing

Feel free to contribute by:
- Reporting bugs or issues
- Suggesting improvements
- Adding new features
- Improving documentation

## 📄 License

This project is available for academic and research purposes. 

## 🔍 Patch-Based Processing

### Why Use Patches?

For high-resolution UAV images (e.g., 5280×3956), patch-based processing offers:

- **Preserved Detail**: 100% resolution vs 87% loss from downsampling
- **Data Multiplication**: 45 images → 1,500+ patches for training
- **Better Small Object Detection**: Weeds become 50-200px instead of 10-50px
- **Improved Performance**: Expected Dice improvement from 0.40 → 0.65-0.75

### Patch Configuration

```json
{
  "patches": {
    "enabled": true,           // Enable/disable patch extraction
    "patch_size": 512,         // Size of square patches
    "overlap": 64,             // Overlap between patches (pixels)
    "balance_strategy": "weed_focused",  // "weed_focused", "balanced", "all_weed"
    "min_weed_pixels": 10,     // Minimum weed pixels to include patch
    "max_patches_per_image": null  // Limit patches per image (null = no limit)
  }
}
```

### Balance Strategies

- **`weed_focused`**: 70% patches with weeds, 30% background
- **`balanced`**: 50% patches with weeds, 50% background  
- **`all_weed`**: Only patches containing weeds

### Patch-Based Inference

When testing models trained with patches, the system automatically:

1. **Detects Patch Training**: Reads config to check if patches were used
2. **Extracts Test Patches**: Splits test images into overlapping patches
3. **Batch Prediction**: Processes patches efficiently with progress updates
4. **Reconstruction**: Combines patch predictions with weighted blending
5. **Full Resolution Output**: Returns predictions at original image resolution

**Key Benefits:**
- Maintains training/testing consistency
- Preserves full image resolution
- Handles memory efficiently
- Provides smooth prediction blending

### Example Usage

```python
from test_model import ModelTester

# Initialize with config (auto-detects patch settings)
tester = ModelTester("best_model.pth", "config.json")

# Test high-resolution image with automatic patching
prediction, probabilities, _, original = tester.predict("high_res_image.jpg")

print(f"Original image: {original.shape}")      # e.g., (3956, 5280, 3)
print(f"Prediction: {prediction.shape}")        # e.g., (3956, 5280)
``` 