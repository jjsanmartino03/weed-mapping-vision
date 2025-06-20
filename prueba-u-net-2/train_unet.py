import os
import json
import time
import argparse
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from roboflow import Roboflow

# Import our custom modules
from unet_model import UNet
from dataset import create_data_loaders
from losses import BCEDiceLoss, dice_coefficient, iou_score
from data_utils import split_dataset, create_directories
from visualization import save_sample_predictions, plot_training_history

class UNetTrainer:
    def __init__(self, config):
        self.config = config
        # Proper device detection for Apple Silicon, CUDA, and CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = config['output_dir']
        create_directories(self.output_dir)
        
        # Initialize model
        self.model = UNet(n_channels=3, n_classes=1, bilinear=config.get('bilinear', False))
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = BCEDiceLoss(
            bce_weight=config.get('bce_weight', 0.5),
            dice_weight=config.get('dice_weight', 0.5)
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        self.train_ious = []
        self.val_ious = []
        
        self.best_val_dice = 0.0
        self.best_model_path = os.path.join(self.output_dir, 'checkpoints', 'best_model.pth')
    
    def download_dataset(self):
        """Download dataset from Roboflow"""
        print("Downloading dataset from Roboflow...")
        
        rf = Roboflow(api_key=self.config['roboflow_api_key'])
        project = rf.workspace(self.config['workspace']).project(self.config['project'])
        version = project.version(self.config['version'])
        
        # Download dataset
        dataset = version.download("yolov8", location=self.config['raw_data_dir'])
        
        print(f"Dataset downloaded to: {self.config['raw_data_dir']}")
        return dataset
    
    def prepare_dataset(self):
        """Process and split the dataset"""
        print("Preparing dataset...")
        
        # Paths to raw data
        raw_data_dir = self.config['raw_data_dir']
        train_images_dir = os.path.join(raw_data_dir, 'train', 'images')
        train_labels_dir = os.path.join(raw_data_dir, 'train', 'labels')
        
        if not os.path.exists(train_images_dir):
            print("Raw dataset not found. Downloading...")
            self.download_dataset()
        
        # Split dataset and create masks
        split_info = split_dataset(
            images_dir=train_images_dir,
            labels_dir=train_labels_dir,
            output_base_dir=self.config['processed_data_dir'],
            train_ratio=self.config.get('train_ratio', 0.7),
            val_ratio=self.config.get('val_ratio', 0.2),
            test_ratio=self.config.get('test_ratio', 0.1)
        )
        
        print("Dataset preparation completed!")
        return split_info
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        num_batches = len(data_loader)
        
        progress_bar = tqdm(data_loader, desc="Training")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)
                iou = iou_score(outputs, masks)
            
            # Update running averages
            epoch_loss += loss.item()
            epoch_dice += dice.item()
            epoch_iou += iou.item()
            
            # Clear cache to free memory (important for MPS)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'IoU': f'{iou.item():.4f}'
            })
        
        # Calculate epoch averages
        epoch_loss /= num_batches
        epoch_dice /= num_batches
        epoch_iou /= num_batches
        
        return epoch_loss, epoch_dice, epoch_iou
    
    def validate_epoch(self, data_loader):
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        num_batches = len(data_loader)
        
        progress_bar = tqdm(data_loader, desc="Validation")
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(progress_bar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                dice = dice_coefficient(outputs, masks)
                iou = iou_score(outputs, masks)
                
                # Update running averages
                epoch_loss += loss.item()
                epoch_dice += dice.item()
                epoch_iou += iou.item()
                
                # Clear cache to free memory (important for MPS)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.item():.4f}',
                    'IoU': f'{iou.item():.4f}'
                })
        
        # Calculate epoch averages
        epoch_loss /= num_batches
        epoch_dice /= num_batches
        epoch_iou /= num_batches
        
        return epoch_loss, epoch_dice, epoch_iou
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"New best model saved! Validation Dice: {self.best_val_dice:.4f}")
    
    def train(self, data_loaders):
        """Main training loop"""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_dice, train_iou = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dices.append(train_dice)
            self.val_dices.append(val_dice)
            self.train_ious.append(train_iou)
            self.val_ious.append(val_iou)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Dice/Train', train_dice, epoch)
            self.writer.add_scalar('Dice/Val', val_dice, epoch)
            self.writer.add_scalar('IoU/Train', train_iou, epoch)
            self.writer.add_scalar('IoU/Val', val_iou, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save checkpoint
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
            
            self.save_checkpoint(epoch + 1, is_best)
            
            # Save sample predictions every few epochs
            if epoch == 1 or ((epoch + 1) % 5 == 0):
                save_sample_predictions(
                    self.model, val_loader, self.device,
                    save_path=os.path.join(self.output_dir, 'results', f'predictions_epoch_{epoch+1}.png'),
                    num_samples=4
                )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")
        
        # Plot and save training history
        plot_training_history(
            self.train_losses, self.val_losses,
            self.train_dices, self.val_dices,
            self.train_ious, self.val_ious,
            save_path=os.path.join(self.output_dir, 'results', 'training_history.png')
        )
        
        # Save training metrics
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious,
            'best_val_dice': self.best_val_dice,
            'training_time_hours': training_time / 3600
        }
        
        with open(os.path.join(self.output_dir, 'results', 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train U-Net for weed segmentation')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--api_key', type=str, required=True, help='Roboflow API key')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'roboflow_api_key': args.api_key,
        'workspace': 'proyecto-finalmapeo-de-malezas',
        'project': 'identificacion-malezas',
        'version': 4,
        'raw_data_dir': 'raw_dataset',
        'processed_data_dir': 'processed_dataset',
        'output_dir': 'training_output',
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'image_size': (1024, 1024),
        'num_workers': 4,
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
        'bce_weight': 0.5,
        'dice_weight': 0.5,
        'bilinear': False
    }
    
    # Load config file if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Initialize trainer
    trainer = UNetTrainer(config)
    
    # Prepare dataset
    split_info = trainer.prepare_dataset()
    
    # Create data loaders
    data_loaders, datasets = create_data_loaders(
        config['processed_data_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers']
    )
    
    # Print dataset info
    print("\nDataset Information:")
    for split, loader in data_loaders.items():
        print(f"{split.capitalize()}: {len(datasets[split])} samples, {len(loader)} batches")
    
    # Start training
    trainer.train(data_loaders)

if __name__ == "__main__":
    main() 