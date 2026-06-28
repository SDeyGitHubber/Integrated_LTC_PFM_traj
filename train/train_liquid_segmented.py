"""
Training Script for Linear/Non-Linear Trajectory Segmentation

This script:
1. Classifies trajectories as linear vs non-linear using regression fitting
2. Trains separate models (or tracks separate losses) for each segment
3. Uses UN-REGULARIZED model for better non-linear learning

Key Innovation:
--------------
Traditional trajectory models struggle with diverse motion patterns.
By separating linear (straight-line) and non-linear (curved/complex)
trajectories, we can:
- Better evaluate model performance on different motion types
- Apply different training strategies to each segment
- Understand where the model excels vs struggles
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from models.liquid_velocity_model import LiquidVelocityModel
from utils.trajectory_linearity_classifier import (
    TrajectoryLinearityClassifier,
    split_batch_by_linearity,
    visualize_linearity_classification
)


class SegmentedMetricsTracker:
    """
    Tracks metrics separately for linear and non-linear segments.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.linear_loss = 0.0
        self.nonlinear_loss = 0.0
        self.linear_count = 0
        self.nonlinear_count = 0
        self.total_linear_samples = 0
        self.total_nonlinear_samples = 0
    
    def update(self, linear_loss, nonlinear_loss, num_linear, num_nonlinear):
        """Update metrics with batch results."""
        if num_linear > 0:
            self.linear_loss += linear_loss * num_linear
            self.linear_count += 1
            self.total_linear_samples += num_linear
        
        if num_nonlinear > 0:
            self.nonlinear_loss += nonlinear_loss * num_nonlinear
            self.nonlinear_count += 1
            self.total_nonlinear_samples += num_nonlinear
    
    def get_averages(self):
        """Get average losses for each segment."""
        avg_linear = self.linear_loss / max(self.total_linear_samples, 1)
        avg_nonlinear = self.nonlinear_loss / max(self.total_nonlinear_samples, 1)
        
        return {
            'linear_loss': avg_linear,
            'nonlinear_loss': avg_nonlinear,
            'linear_samples': self.total_linear_samples,
            'nonlinear_samples': self.total_nonlinear_samples,
            'linear_ratio': self.total_linear_samples / max(self.total_linear_samples + self.total_nonlinear_samples, 1)
        }


def velocity_residual_loss(velocities, reduction='mean'):
    """
    Residual velocity loss (regularization).
    Penalizes deviations from current velocity.
    
    Args:
        velocities: Tensor (B, A, T, 2) - predicted velocities [v, omega]
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: Scalar tensor
    """
    residual_norm = torch.sqrt((velocities ** 2).sum(dim=-1) + 1e-8)
    
    if reduction == 'mean':
        return residual_norm.mean()
    elif reduction == 'sum':
        return residual_norm.sum()
    else:
        return residual_norm


def train_one_epoch_segmented(model, train_loader, optimizer, device, config, 
                              classifier, metrics_tracker, epoch):
    """
    Train for one epoch with linear/non-linear segmentation.
    
    Args:
        model: LiquidVelocityModel instance
        train_loader: DataLoader
        optimizer: Optimizer
        device: Device
        config: Configuration dict
        classifier: TrajectoryLinearityClassifier
        metrics_tracker: SegmentedMetricsTracker
        epoch: Current epoch number
    
    Returns:
        metrics: Dict of training metrics
    """
    model.train()
    metrics_tracker.reset()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training", ncols=140)
    
    for batch_idx, (hist_neighbors, future, _, _, exp_goals, _) in enumerate(pbar):
        hist_neighbors = hist_neighbors.to(device)
        future = future.to(device)
        
        B, A, _, H, _ = hist_neighbors.shape
        
        # Classify trajectories as linear or non-linear
        history_ego = hist_neighbors[:, :, 0, :, :]  # (B, A, H, 2)
        classifications, mse_values, regression_lines, threshold = classifier.classify_batch(
            history_ego, future
        )
        
        # Get masks
        linear_mask = classifications == 1
        nonlinear_mask = classifications == 0
        
        num_linear = linear_mask.sum().item()
        num_nonlinear = nonlinear_mask.sum().item()
        
        # Forward pass (full batch)
        optimizer.zero_grad()
        predictions, velocities, _ = model(hist_neighbors, exp_goals)
        
        # Compute losses for each segment
        linear_loss_val = 0.0
        nonlinear_loss_val = 0.0
        total_loss = 0.0
        
        if num_linear > 0:
            # Extract linear trajectories
            linear_vels = velocities[linear_mask]
            linear_loss_val = velocity_residual_loss(linear_vels)
            total_loss += linear_loss_val * num_linear
        
        if num_nonlinear > 0:
            # Extract non-linear trajectories
            nonlinear_vels = velocities[nonlinear_mask]
            nonlinear_loss_val = velocity_residual_loss(nonlinear_vels)
            total_loss += nonlinear_loss_val * num_nonlinear
        
        # Normalize loss by total samples
        if num_linear + num_nonlinear > 0:
            total_loss = total_loss / (num_linear + num_nonlinear)
        
        # Backward pass
        total_loss.backward()
        
        if config['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        
        optimizer.step()
        
        # Update metrics
        metrics_tracker.update(
            linear_loss_val.item() if num_linear > 0 else 0.0,
            nonlinear_loss_val.item() if num_nonlinear > 0 else 0.0,
            num_linear,
            num_nonlinear
        )
        
        # Update progress bar
        current_metrics = metrics_tracker.get_averages()
        pbar.set_postfix({
            'L_loss': f"{current_metrics['linear_loss']:.4f}",
            'NL_loss': f"{current_metrics['nonlinear_loss']:.4f}",
            'L_ratio': f"{current_metrics['linear_ratio']:.2f}",
            'thresh': f"{threshold:.4f}"
        })
    
    return metrics_tracker.get_averages()


def validate_segmented(model, val_loader, device, config, classifier):
    """
    Validate with linear/non-linear segmentation.
    
    Returns:
        metrics: Dict with losses for each segment
    """
    model.eval()
    metrics_tracker = SegmentedMetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", ncols=140)
        
        for hist_neighbors, future, _, _, exp_goals, _ in pbar:
            hist_neighbors = hist_neighbors.to(device)
            future = future.to(device)
            
            # Classify
            history_ego = hist_neighbors[:, :, 0, :, :]
            classifications, mse_values, regression_lines, threshold = classifier.classify_batch(
                history_ego, future
            )
            
            linear_mask = classifications == 1
            nonlinear_mask = classifications == 0
            
            num_linear = linear_mask.sum().item()
            num_nonlinear = nonlinear_mask.sum().item()
            
            # Forward pass
            predictions, velocities, _ = model(hist_neighbors, exp_goals)
            
            # Compute losses
            linear_loss_val = 0.0
            nonlinear_loss_val = 0.0
            
            if num_linear > 0:
                linear_vels = velocities[linear_mask]
                linear_loss_val = velocity_residual_loss(linear_vels).item()
            
            if num_nonlinear > 0:
                nonlinear_vels = velocities[nonlinear_mask]
                nonlinear_loss_val = velocity_residual_loss(nonlinear_vels).item()
            
            # Update metrics
            metrics_tracker.update(linear_loss_val, nonlinear_loss_val, num_linear, num_nonlinear)
            
            current_metrics = metrics_tracker.get_averages()
            pbar.set_postfix({
                'L_loss': f"{current_metrics['linear_loss']:.4f}",
                'NL_loss': f"{current_metrics['nonlinear_loss']:.4f}",
                'L_ratio': f"{current_metrics['linear_ratio']:.2f}"
            })
    
    return metrics_tracker.get_averages()


def train_segmented_model():
    """
    Main training function with linear/non-linear segmentation.
    """
    
    CONFIG = {
        # Data
        'data_path': 'data/combined_annotations.csv',
        'checkpoint_dir': 'checkpoints/liquid_segmented',
        'log_dir': 'logs/liquid_segmented',
        'train_split': 0.8,
        'random_seed': 42,
        
        # Model
        'hidden_size': 64,
        'dt': 0.25,
        'dense_dt': 0.05,
        'prediction_len': 12,
        'max_neighbors': 4,
        'backbone_layers': 1,
        'backbone_units': 64,
        'backbone_dropout': 0.1,
        'activation': 'hardtanh',
        
        # Dataset
        'history_len': 8,
        
        # Training
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'gradient_clip': 1.0,
        
        # Linearity classification
        'linearity_threshold_method': 'adaptive',  # 'adaptive' or 'fixed'
        'linearity_k_std': 1.0,  # For adaptive: mean + k*std
        'linearity_fixed_threshold': 0.1,  # For fixed method
        
        # Scheduler
        'use_scheduler': True,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'scheduler_min_lr': 1e-6,
        
        # Checkpointing
        'save_interval': 5,
        'visualize_interval': 10,  # Visualize every N epochs
    }
    
    # Setup
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    os.makedirs(os.path.join(CONFIG['log_dir'], 'visualizations'), exist_ok=True)
    
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Data loading
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    full_dataset = PFM_TrajectoryDataset_neighbours(
        CONFIG['data_path'],
        history_len=CONFIG['history_len'],
        prediction_len=CONFIG['prediction_len'],
        max_neighbors=CONFIG['max_neighbors']
    )
    
    total_samples = len(full_dataset)
    train_size = int(CONFIG['train_split'] * total_samples)
    val_size = total_samples - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, CONFIG['history_len'], CONFIG['prediction_len'], CONFIG['max_neighbors']),
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, CONFIG['history_len'], CONFIG['prediction_len'], CONFIG['max_neighbors']),
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}\n")
    
    # Initialize linearity classifier
    classifier = TrajectoryLinearityClassifier(
        threshold_method=CONFIG['linearity_threshold_method'],
        threshold_value=CONFIG['linearity_fixed_threshold'],
        k_std=CONFIG['linearity_k_std']
    )
    
    # Model
    print("="*80)
    print("INITIALIZING LIQUID VELOCITY MODEL")
    print("="*80)
    
    model = LiquidVelocityModel(
        hidden_size=CONFIG['hidden_size'],
        dt=CONFIG['dt'],
        dense_dt=CONFIG['dense_dt'],
        prediction_len=CONFIG['prediction_len'],
        max_neighbors=CONFIG['max_neighbors'],
        backbone_layers=CONFIG['backbone_layers'],
        backbone_units=CONFIG['backbone_units'],
        backbone_dropout=CONFIG['backbone_dropout'],
        activation=CONFIG['activation']
    )
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    scheduler = None
    if CONFIG['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=CONFIG['scheduler_factor'],
            patience=CONFIG['scheduler_patience'],
            min_lr=CONFIG['scheduler_min_lr'],
            verbose=True
        )
    
    # Training loop
    print("="*80)
    print("STARTING SEGMENTED TRAINING")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    history = {
        'train_linear_loss': [],
        'train_nonlinear_loss': [],
        'val_linear_loss': [],
        'val_nonlinear_loss': [],
        'linear_ratio': [],
        'epochs': []
    }
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        metrics_tracker = SegmentedMetricsTracker()
        
        # Train
        train_metrics = train_one_epoch_segmented(
            model, train_loader, optimizer, device, CONFIG, classifier, metrics_tracker, epoch
        )
        
        # Validate
        val_metrics = validate_segmented(model, val_loader, device, CONFIG, classifier)
        
        # Update scheduler
        if scheduler:
            combined_val_loss = (val_metrics['linear_loss'] + val_metrics['nonlinear_loss']) / 2
            scheduler.step(combined_val_loss)
        
        # Track history
        history['train_linear_loss'].append(train_metrics['linear_loss'])
        history['train_nonlinear_loss'].append(train_metrics['nonlinear_loss'])
        history['val_linear_loss'].append(val_metrics['linear_loss'])
        history['val_nonlinear_loss'].append(val_metrics['nonlinear_loss'])
        history['linear_ratio'].append(val_metrics['linear_ratio'])
        history['epochs'].append(epoch)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*80}")
        print(f"Train - Linear: {train_metrics['linear_loss']:.6f} | Non-Linear: {train_metrics['nonlinear_loss']:.6f}")
        print(f"Val   - Linear: {val_metrics['linear_loss']:.6f} | Non-Linear: {val_metrics['nonlinear_loss']:.6f}")
        print(f"Linear Ratio: {val_metrics['linear_ratio']:.2%} | Threshold: {classifier.computed_threshold:.6f}")
        
        # Save checkpoint
        if epoch % CONFIG['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
                'history': history,
                'linearity_threshold': classifier.computed_threshold
            }
            
            save_path = os.path.join(CONFIG['checkpoint_dir'], f"epoch_{epoch:03d}.pth")
            torch.save(checkpoint, save_path)
            print(f"✓ Saved checkpoint: {save_path}")
        
        # Save history
        with open(os.path.join(CONFIG['log_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    train_segmented_model()
