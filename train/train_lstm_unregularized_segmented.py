"""
Training Script for Un-Regularized LSTM with Linear/Non-Linear Segmentation

This script trains the UNREGULARIZED VelocityBasedLSTM model on both
linear and non-linear trajectory segments, tracking performance separately.

Key Features:
------------
1. Automatic trajectory classification (linear vs non-linear)
2. Un-regularized LSTM (NO residual bias, NO speed clamping)
3. Position MSE loss (NOT residual minimization)
4. Separate metrics tracking for each segment
5. Single model trained on both types
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from models.lstm_ang_vel_unregularized import VelocityBasedLSTMUnregularized
from utils.trajectory_linearity_classifier import (
    TrajectoryLinearityClassifier,
    visualize_linearity_classification
)


class SegmentedMetricsTracker:
    """Tracks metrics separately for linear and non-linear segments."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.linear_loss = 0.0
        self.nonlinear_loss = 0.0
        self.linear_count = 0
        self.nonlinear_count = 0
        self.total_linear_samples = 0
        self.total_nonlinear_samples = 0
        self.batch_losses = []
    
    def update(self, linear_loss, nonlinear_loss, num_linear, num_nonlinear):
        if num_linear > 0:
            self.linear_loss += linear_loss * num_linear
            self.linear_count += 1
            self.total_linear_samples += num_linear
        
        if num_nonlinear > 0:
            self.nonlinear_loss += nonlinear_loss * num_nonlinear
            self.nonlinear_count += 1
            self.total_nonlinear_samples += num_nonlinear
        
        total_loss = (linear_loss * num_linear + nonlinear_loss * num_nonlinear) / max(num_linear + num_nonlinear, 1)
        self.batch_losses.append(total_loss)
    
    def get_averages(self):
        avg_linear = self.linear_loss / max(self.total_linear_samples, 1)
        avg_nonlinear = self.nonlinear_loss / max(self.total_nonlinear_samples, 1)
        avg_combined = (self.linear_loss + self.nonlinear_loss) / max(self.total_linear_samples + self.total_nonlinear_samples, 1)
        
        return {
            'linear_loss': avg_linear,
            'nonlinear_loss': avg_nonlinear,
            'combined_loss': avg_combined,
            'linear_samples': self.total_linear_samples,
            'nonlinear_samples': self.total_nonlinear_samples,
            'linear_ratio': self.total_linear_samples / max(self.total_linear_samples + self.total_nonlinear_samples, 1)
        }


def position_mse_loss(predictions, targets):
    """
    Direct MSE loss on positions (NO regularization).
    
    This is the KEY difference from the regularized version:
    - NO residual norm minimization
    - Direct position accuracy focus
    """
    return ((predictions - targets) ** 2).sum(dim=-1).mean()


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                improvement = self.best_loss - val_loss
                print(f"[EarlyStopping] Improved by {improvement:.6f}")
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch_segmented(model, train_loader, optimizer, device, config, 
                              classifier, metrics_tracker, epoch):
    """
    Train for one epoch with linear/non-linear segmentation.
    """
    model.train()
    metrics_tracker.reset()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training", ncols=140)
    
    for batch_idx, (hist_neighbors, future, _, _, exp_goals, _) in enumerate(pbar):
        hist_neighbors = hist_neighbors.to(device)
        future = future.to(device)
        
        # Classify trajectories
        history_ego = hist_neighbors[:, :, 0, :, :]
        classifications, mse_values, regression_lines, threshold = classifier.classify_batch(
            history_ego, future
        )
        
        linear_mask = classifications == 1
        nonlinear_mask = classifications == 0
        
        num_linear = linear_mask.sum().item()
        num_nonlinear = nonlinear_mask.sum().item()
        
        # Forward pass
        optimizer.zero_grad()
        predictions, velocities, _ = model(hist_neighbors, exp_goals)
        
        # Compute losses for each segment
        linear_loss_val = 0.0
        nonlinear_loss_val = 0.0
        total_loss = 0.0
        
        if num_linear > 0:
            linear_preds = predictions[linear_mask]
            linear_targets = future[linear_mask]
            linear_loss_val = position_mse_loss(linear_preds, linear_targets)
            total_loss += linear_loss_val * num_linear
        
        if num_nonlinear > 0:
            nonlinear_preds = predictions[nonlinear_mask]
            nonlinear_targets = future[nonlinear_mask]
            nonlinear_loss_val = position_mse_loss(nonlinear_preds, nonlinear_targets)
            total_loss += nonlinear_loss_val * num_nonlinear
        
        # Normalize
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
            'L': f"{current_metrics['linear_loss']:.4f}",
            'NL': f"{current_metrics['nonlinear_loss']:.4f}",
            'ratio': f"{current_metrics['linear_ratio']:.2f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
        })
    
    return metrics_tracker.get_averages()


def validate_segmented(model, val_loader, device, classifier):
    """Validate with segmentation."""
    model.eval()
    metrics_tracker = SegmentedMetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", ncols=140)
        
        for hist_neighbors, future, _, _, exp_goals, _ in pbar:
            hist_neighbors = hist_neighbors.to(device)
            future = future.to(device)
            
            history_ego = hist_neighbors[:, :, 0, :, :]
            classifications, mse_values, regression_lines, threshold = classifier.classify_batch(
                history_ego, future
            )
            
            linear_mask = classifications == 1
            nonlinear_mask = classifications == 0
            
            num_linear = linear_mask.sum().item()
            num_nonlinear = nonlinear_mask.sum().item()
            
            predictions, velocities, _ = model(hist_neighbors, exp_goals)
            
            linear_loss_val = 0.0
            nonlinear_loss_val = 0.0
            
            if num_linear > 0:
                linear_preds = predictions[linear_mask]
                linear_targets = future[linear_mask]
                linear_loss_val = position_mse_loss(linear_preds, linear_targets).item()
            
            if num_nonlinear > 0:
                nonlinear_preds = predictions[nonlinear_mask]
                nonlinear_targets = future[nonlinear_mask]
                nonlinear_loss_val = position_mse_loss(nonlinear_preds, nonlinear_targets).item()
            
            metrics_tracker.update(linear_loss_val, nonlinear_loss_val, num_linear, num_nonlinear)
            
            current_metrics = metrics_tracker.get_averages()
            pbar.set_postfix({
                'L': f"{current_metrics['linear_loss']:.4f}",
                'NL': f"{current_metrics['nonlinear_loss']:.4f}"
            })
    
    return metrics_tracker.get_averages()


def train_lstm_unregularized_segmented():
    """
    Main training function for un-regularized LSTM with segmentation.
    """
    
    CONFIG = {
        # Data
        'data_path': 'data/combined_annotations.csv',
        'checkpoint_dir': 'checkpoints/lstm_unregularized_segmented',
        'log_dir': 'logs/lstm_unregularized_segmented',
        'train_split': 0.8,
        'random_seed': 42,
        
        # Model (UN-REGULARIZED LSTM)
        'hidden_size': 64,
        'num_layers': 2,
        'dt': 0.25,
        'output_mode': 'velocities',    # 'velocities' or 'positions'
        'use_speed_limits': False,       # NO speed clamping (key de-regularization)
        'max_speed': 10.0,               # Only if use_speed_limits=True
        
        # Dataset
        'history_len': 8,
        'prediction_len': 12,
        'max_neighbors': 4,
        
        # Training
        'batch_size': 16,
        'learning_rate': 1e-3,           # Higher LR since no regularization
        'num_epochs': 50,
        'gradient_clip': 1.0,
        
        # Linearity classification
        'linearity_threshold_method': 'adaptive',
        'linearity_k_std': 1.0,
        
        # Scheduler
        'use_scheduler': True,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'scheduler_min_lr': 1e-6,
        
        # Early stopping
        'use_early_stopping': True,
        'early_stopping_patience': 10,
        
        # Checkpointing
        'save_interval': 5,
    }
    
    # Setup
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("UN-REGULARIZED LSTM TRAINING WITH LINEAR/NON-LINEAR SEGMENTATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Output mode: {CONFIG['output_mode']}")
    print(f"Speed limits: {CONFIG['use_speed_limits']}")
    print("="*80 + "\n")
    
    # Data loading
    print("Loading dataset...")
    
    # Try multiple paths
    possible_paths = [
        CONFIG['data_path'],
        '/content/combined_annotations.csv',
        'combined_annotations.csv',
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Data file not found. Tried: {possible_paths}")
    
    print(f"Using data: {data_path}")
    
    full_dataset = PFM_TrajectoryDataset_neighbours(
        data_path,
        history_len=CONFIG['history_len'],
        prediction_len=CONFIG['prediction_len'],
        max_neighbors=CONFIG['max_neighbors']
    )
    
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty!")
    
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
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, CONFIG['history_len'], CONFIG['prediction_len'], CONFIG['max_neighbors']),
        num_workers=0
    )
    
    print(f"Train: {train_size}, Val: {val_size}, Total: {total_samples}\n")
    
    # Linearity classifier
    classifier = TrajectoryLinearityClassifier(
        threshold_method=CONFIG['linearity_threshold_method'],
        k_std=CONFIG['linearity_k_std']
    )
    
    # Model
    print("Initializing UN-REGULARIZED LSTM...")
    
    model = VelocityBasedLSTMUnregularized(
        input_size=2,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dt=CONFIG['dt'],
        output_mode=CONFIG['output_mode'],
        use_speed_limits=CONFIG['use_speed_limits'],
        max_speed=CONFIG['max_speed']
    )
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")
    
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
    
    early_stopping = None
    if CONFIG['use_early_stopping']:
        early_stopping = EarlyStopping(
            patience=CONFIG['early_stopping_patience'],
            verbose=True
        )
    
    # Training loop
    print("="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    history = {
        'train_linear': [],
        'train_nonlinear': [],
        'val_linear': [],
        'val_nonlinear': [],
        'linear_ratio': [],
        'threshold': [],
        'epochs': []
    }
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        metrics_tracker = SegmentedMetricsTracker()
        
        # Train
        train_metrics = train_one_epoch_segmented(
            model, train_loader, optimizer, device, CONFIG, 
            classifier, metrics_tracker, epoch
        )
        
        # Validate
        val_metrics = validate_segmented(model, val_loader, device, classifier)
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_metrics['combined_loss'])
        
        # Track history
        history['train_linear'].append(train_metrics['linear_loss'])
        history['train_nonlinear'].append(train_metrics['nonlinear_loss'])
        history['val_linear'].append(val_metrics['linear_loss'])
        history['val_nonlinear'].append(val_metrics['nonlinear_loss'])
        history['linear_ratio'].append(val_metrics['linear_ratio'])
        history['threshold'].append(classifier.computed_threshold)
        history['epochs'].append(epoch)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{CONFIG['num_epochs']}")
        print(f"{'='*80}")
        print(f"Train  → Linear: {train_metrics['linear_loss']:.6f} | Non-Linear: {train_metrics['nonlinear_loss']:.6f}")
        print(f"Val    → Linear: {val_metrics['linear_loss']:.6f} | Non-Linear: {val_metrics['nonlinear_loss']:.6f}")
        print(f"Linear Ratio: {val_metrics['linear_ratio']:.2%} | Threshold: {classifier.computed_threshold:.6f}")
        print(f"Combined Val Loss: {val_metrics['combined_loss']:.6f}")
        
        # Check if best
        is_best = val_metrics['combined_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['combined_loss']
            print(f"✓ NEW BEST MODEL")
        
        # Save checkpoint
        if epoch % CONFIG['save_interval'] == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
                'history': history,
                'best_val_loss': best_val_loss
            }
            
            save_path = os.path.join(CONFIG['checkpoint_dir'], f"epoch_{epoch:03d}.pth")
            torch.save(checkpoint, save_path)
            
            if is_best:
                best_path = os.path.join(CONFIG['checkpoint_dir'], "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f"✓ Saved: {save_path}")
        
        # Save history
        with open(os.path.join(CONFIG['log_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        # Early stopping
        if early_stopping:
            early_stopping(val_metrics['combined_loss'])
            if early_stopping.early_stop:
                print(f"\nEARLY STOPPING at epoch {epoch}")
                break
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {CONFIG['checkpoint_dir']}")
    print(f"Logs: {CONFIG['log_dir']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        train_lstm_unregularized_segmented()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
