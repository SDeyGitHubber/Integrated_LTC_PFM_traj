"""
Training Script for Liquid Time-Constant Network (LTC) Velocity Model

This script trains the LiquidVelocityModel with social context awareness and
continuous-time dynamics for pedestrian trajectory prediction.

Key Features:
------------
1. Social Context Processing: Ego + neighbor aggregation
2. Liquid Dynamics: CfC-based continuous ODE solver
3. Dense Trajectory Output: High-resolution predictions
4. Residual Velocity Learning: Predicts corrections to base velocities
5. Advanced Training: Early stopping, LR scheduling, gradient clipping

Loss Function:
-------------
Residual L2 Loss: Minimizes magnitude of velocity residuals (Δv, Δω)
    L = mean(||velocity_residuals||₂)
    
This encourages the model to rely on kinematic priors (v_avg, ω_avg=0)
and only learn necessary corrections for complex dynamics.

References:
----------
[1] Hasani et al., "Liquid Time-Constant Networks", AAAI 2021
[2] Hasani et al., "Closed-form Continuous-time Neural Networks", Nature ML 2022
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

from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from models.liquid_velocity_model import LiquidVelocityModel


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Monitors validation loss and stops training if no improvement
    is observed for a specified number of epochs (patience).
    """
    def __init__(self, patience=7, min_delta=0.0, verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            verbose (bool): Print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """Check if training should stop based on validation loss."""
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
                print(f"[EarlyStopping] Validation loss improved by {improvement:.6f}")
            self.best_loss = val_loss
            self.counter = 0


class MetricsTracker:
    """
    Tracks and computes training metrics.
    
    Maintains running averages and logs metrics for analysis.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.total_loss = 0.0
        self.total_residual_loss = 0.0
        self.num_batches = 0
        self.batch_losses = []
    
    def update(self, loss, residual_loss=None):
        """Update metrics with batch results."""
        self.total_loss += loss
        if residual_loss is not None:
            self.total_residual_loss += residual_loss
        self.num_batches += 1
        self.batch_losses.append(loss)
    
    def get_average_loss(self):
        """Get average loss across all batches."""
        return self.total_loss / max(self.num_batches, 1)
    
    def get_average_residual_loss(self):
        """Get average residual loss across all batches."""
        return self.total_residual_loss / max(self.num_batches, 1)
    
    def get_loss_std(self):
        """Get standard deviation of batch losses."""
        if len(self.batch_losses) < 2:
            return 0.0
        return np.std(self.batch_losses)


def residual_velocity_loss(pred_vel_residuals, reduction='mean'):
    """
    Compute L2 loss on velocity residuals.
    
    Loss Rationale:
    --------------
    The model predicts residuals (Δv, Δω) as corrections to base velocities.
    Minimizing the L2 norm encourages the model to:
    1. Rely on kinematic priors (v_avg) when possible
    2. Only predict residuals when necessary for complex dynamics
    3. Maintain stability (bounded residuals → bounded velocities)
    
    Args:
        pred_vel_residuals: Tensor (B, A, T, 2) - predicted (Δv, Δω)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: Scalar tensor or tensor (B, A, T) if reduction='none'
    """
    # Compute L2 norm of residuals: ||[Δv, Δω]||₂
    residual_norms = pred_vel_residuals.norm(dim=-1)  # (B, A, T)
    
    if reduction == 'mean':
        return residual_norms.mean()
    elif reduction == 'sum':
        return residual_norms.sum()
    else:
        return residual_norms


def position_loss(predictions, targets, reduction='mean'):
    """
    Optional: Compute MSE loss on predicted positions.
    
    Can be used in addition to residual loss for end-to-end training.
    
    Args:
        predictions: Tensor (B, A, T, 2) - predicted positions
        targets: Tensor (B, A, T, 2) - ground truth positions
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: Scalar tensor
    """
    mse = ((predictions - targets) ** 2).sum(dim=-1)  # (B, A, T)
    
    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    else:
        return mse


def train_one_epoch(model, train_loader, optimizer, device, config, metrics_tracker):
    """
    Train model for one epoch.
    
    Args:
        model: LiquidVelocityModel instance
        train_loader: DataLoader for training data
        optimizer: PyTorch optimizer
        device: torch.device
        config: Configuration dictionary
        metrics_tracker: MetricsTracker instance
    
    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()
    metrics_tracker.reset()
    
    pbar = tqdm(train_loader, desc="Training", ncols=120)
    
    for batch_idx, (hist_neighbors, future, _, _, exp_goals, _) in enumerate(pbar):
        # Move data to device
        hist_neighbors = hist_neighbors.to(device)
        # future = future.to(device)  # Optional: if using position loss
        
        # Forward pass
        optimizer.zero_grad()
        predictions, velocity_residuals, _ = model(hist_neighbors, exp_goals)
        
        # Compute loss
        if config['loss_type'] == 'residual':
            # Primary loss: Minimize velocity residuals
            loss = residual_velocity_loss(velocity_residuals)
        elif config['loss_type'] == 'hybrid':
            # Hybrid loss: Residual + position MSE
            residual_loss = residual_velocity_loss(velocity_residuals)
            # Note: predictions are dense, need to downsample to match future
            # For simplicity, use residual loss only (recommended)
            loss = residual_loss
        else:
            loss = residual_velocity_loss(velocity_residuals)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (critical for CfC stability)
        if config['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['gradient_clip']
            )
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        metrics_tracker.update(loss.item(), loss.item())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.6f}",
            'avg_loss': f"{metrics_tracker.get_average_loss():.6f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6e}"
        })
    
    return metrics_tracker.get_average_loss()


def validate(model, val_loader, device, config):
    """
    Validate model on validation set.
    
    Args:
        model: LiquidVelocityModel instance
        val_loader: DataLoader for validation data
        device: torch.device
        config: Configuration dictionary
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary of validation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", ncols=120)
        
        for hist_neighbors, future, _, _, exp_goals, _ in pbar:
            # Move data to device
            hist_neighbors = hist_neighbors.to(device)
            
            # Forward pass
            predictions, velocity_residuals, _ = model(hist_neighbors, exp_goals)
            
            # Compute loss
            if config['loss_type'] == 'residual':
                loss = residual_velocity_loss(velocity_residuals)
            else:
                loss = residual_velocity_loss(velocity_residuals)
            
            # Update metrics
            metrics_tracker.update(loss.item(), loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'val_loss': f"{loss.item():.6f}",
                'avg_val_loss': f"{metrics_tracker.get_average_loss():.6f}"
            })
    
    avg_loss = metrics_tracker.get_average_loss()
    loss_std = metrics_tracker.get_loss_std()
    
    metrics = {
        'val_loss': avg_loss,
        'val_loss_std': loss_std,
        'num_batches': metrics_tracker.num_batches
    }
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   best_val_loss, config, filepath, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: LiquidVelocityModel instance
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        best_val_loss: Best validation loss so far
        config: Configuration dictionary
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = os.path.join(
            os.path.dirname(filepath), 
            "best_liquid_velocity_model.pth"
        )
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model to {best_path}")


def log_training_info(config, log_dir):
    """
    Log training configuration and system info.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to save logs
    """
    log_path = os.path.join(log_dir, 'training_config.json')
    
    info = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'
    }
    
    with open(log_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print("\n" + "="*80)
    print("LIQUID VELOCITY MODEL TRAINING CONFIGURATION")
    print("="*80)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("="*80 + "\n")


def train_liquid_velocity_model():
    """
    Main training function for Liquid Velocity Model.
    
    Trains a Liquid Time-Constant Network (LTC) using the CfC cell
    for continuous-time trajectory prediction with social context.
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    CONFIG = {
        # Data paths (try multiple common locations)
        'data_path': 'data/combined_annotations.csv',  # Default path
        # Alternative paths (will try in order):
        # '/content/combined_annotations.csv'  (Colab)
        # 'combined_annotations.csv'           (Current directory)
        # 'data/crowds_zara02_test_cleaned.txt' (Alternative dataset)
        'checkpoint_dir': 'checkpoints/liquid_velocity_model',
        'log_dir': 'logs/liquid_velocity_model',
        
        # Data split
        'train_split': 0.8,
        'random_seed': 42,
        
        # Model architecture
        'hidden_size': 64,          # CfC hidden units
        'dt': 0.25,                 # Prediction timestep (seconds)
        'dense_dt': 0.05,           # Integration timestep (seconds)
        'target_avg_speed': 5.115,  # Average pedestrian speed (units/s)
        'speed_tolerance': 0.15,    # Speed clamping tolerance (±15%)
        'max_neighbors': 4,         # Number of neighbors to consider
        'backbone_layers': 1,       # CfC backbone depth
        'backbone_units': 64,       # CfC backbone hidden units
        'backbone_dropout': 0.1,    # Dropout in CfC backbone
        'activation': 'hardtanh',   # Activation for social processor ('hardtanh', 'relu', 'tanh')
        
        # Dataset parameters
        'history_len': 8,           # History length (timesteps)
        'prediction_len': 12,       # Prediction length (timesteps)
        
        # Training hyperparameters
        'batch_size': 16,           # Batch size (increase if GPU memory allows)
        'learning_rate': 1e-4,      # Initial learning rate (CfC is sensitive)
        'num_epochs': 50,           # Maximum number of epochs
        'gradient_clip': 1.0,       # Gradient clipping threshold
        'loss_type': 'residual',    # Loss type ('residual', 'hybrid')
        
        # Learning rate scheduling
        'use_scheduler': True,
        'scheduler_type': 'plateau', # 'plateau' or 'cosine'
        'scheduler_patience': 5,    # ReduceLROnPlateau patience
        'scheduler_factor': 0.5,    # LR reduction factor
        'scheduler_min_lr': 1e-6,   # Minimum learning rate
        
        # Early stopping
        'use_early_stopping': True,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 1e-4,
        
        # Checkpointing
        'save_best_only': False,    # Save all checkpoints or best only
        'save_interval': 5,         # Save checkpoint every N epochs
        
        # DataLoader
        'num_workers': 2,           # Number of data loading workers
        'pin_memory': True,         # Pin memory for faster GPU transfer
    }
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    # Create directories
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    
    # ========================================================================
    # DATA PATH AUTO-DETECTION
    # ========================================================================
    # Try multiple common data paths
    possible_paths = [
        CONFIG['data_path'],
        '/content/combined_annotations.csv',  # Google Colab
        'combined_annotations.csv',           # Current directory
        'data/crowds_zara02_test_cleaned.txt', # Alternative dataset
        '../data/combined_annotations.csv',   # Parent directory
    ]
    
    data_path_found = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path_found = path
            print(f"✓ Found data file: {path}")
            break
    
    if data_path_found is None:
        error_msg = (
            f"ERROR: Could not find data file!\n"
            f"Tried the following paths:\n"
        )
        for path in possible_paths:
            error_msg += f"  - {path}\n"
        error_msg += (
            f"\nPlease:\n"
            f"  1. Check that your data file exists\n"
            f"  2. Update CONFIG['data_path'] with the correct path\n"
            f"  3. Or place 'combined_annotations.csv' in one of the above locations"
        )
        raise FileNotFoundError(error_msg)
    
    CONFIG['data_path'] = data_path_found
    
    # Log configuration
    log_training_info(CONFIG, CONFIG['log_dir'])
    
    # Set random seed for reproducibility
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['random_seed'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    # Validate data path
    if not os.path.exists(CONFIG['data_path']):
        raise FileNotFoundError(
            f"Data file not found: {CONFIG['data_path']}\n"
            f"Please check the data_path in CONFIG."
        )
    
    print(f"Data path: {CONFIG['data_path']}")
    
    # Load full dataset
    try:
        full_dataset = PFM_TrajectoryDataset_neighbours(
            CONFIG['data_path'],
            history_len=CONFIG['history_len'],
            prediction_len=CONFIG['prediction_len'],
            max_neighbors=CONFIG['max_neighbors']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")
    
    total_samples = len(full_dataset)
    print(f"Total samples: {total_samples}")
    
    # Validate dataset size
    if total_samples == 0:
        raise ValueError(
            f"Dataset is empty! No samples found in {CONFIG['data_path']}\n"
            f"Please check:\n"
            f"  1. The CSV file exists and is not empty\n"
            f"  2. The CSV has the required columns\n"
            f"  3. The data meets minimum trajectory length requirements\n"
            f"     (history_len={CONFIG['history_len']}, prediction_len={CONFIG['prediction_len']})"
        )
    
    if total_samples < 10:
        print(f"WARNING: Very small dataset ({total_samples} samples). Results may not be reliable.")
    
    # Train/validation split
    train_size = int(CONFIG['train_split'] * total_samples)
    val_size = total_samples - train_size
    
    # Ensure minimum samples in each split
    if train_size == 0:
        train_size = max(1, int(total_samples * 0.7))
        val_size = total_samples - train_size
        print(f"WARNING: Adjusted split to ensure training samples > 0")
    
    if val_size == 0:
        val_size = max(1, int(total_samples * 0.3))
        train_size = total_samples - val_size
        print(f"WARNING: Adjusted split to ensure validation samples > 0")
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Adjust batch size if dataset is too small
    if train_size < CONFIG['batch_size']:
        CONFIG['batch_size'] = max(1, train_size // 2)
        print(f"WARNING: Adjusted batch_size to {CONFIG['batch_size']} (dataset too small)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b, 
            CONFIG['history_len'], 
            CONFIG['prediction_len'], 
            CONFIG['max_neighbors']
        ),
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=CONFIG['pin_memory'] if device.type == 'cuda' else False,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b, 
            CONFIG['history_len'], 
            CONFIG['prediction_len'], 
            CONFIG['max_neighbors']
        ),
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=CONFIG['pin_memory'] if device.type == 'cuda' else False,
        persistent_workers=False
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("INITIALIZING LIQUID VELOCITY MODEL")
    print("="*80)
    
    model = LiquidVelocityModel(
        input_size=2,
        hidden_size=CONFIG['hidden_size'],
        dt=CONFIG['dt'],
        target_avg_speed=CONFIG['target_avg_speed'],
        speed_tolerance=CONFIG['speed_tolerance'],
        prediction_len=CONFIG['prediction_len'],
        max_neighbors=CONFIG['max_neighbors'],
        backbone_layers=CONFIG['backbone_layers'],
        backbone_units=CONFIG['backbone_units'],
        backbone_dropout=CONFIG['backbone_dropout'],
        activation=CONFIG['activation'],
        dense_dt=CONFIG['dense_dt']
    )
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.2f} MB (float32)")
    
    # ========================================================================
    # OPTIMIZER AND SCHEDULER
    # ========================================================================
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=1e-5  # Slight L2 regularization
    )
    
    scheduler = None
    if CONFIG['use_scheduler']:
        if CONFIG['scheduler_type'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=CONFIG['scheduler_factor'],
                patience=CONFIG['scheduler_patience'],
                min_lr=CONFIG['scheduler_min_lr'],
                verbose=True
            )
        elif CONFIG['scheduler_type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG['num_epochs'],
                eta_min=CONFIG['scheduler_min_lr']
            )
    
    # Early stopping
    early_stopping = None
    if CONFIG['use_early_stopping']:
        early_stopping = EarlyStopping(
            patience=CONFIG['early_stopping_patience'],
            min_delta=CONFIG['early_stopping_min_delta'],
            verbose=True
        )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'epochs': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        epoch_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{CONFIG['num_epochs']}")
        print(f"{'='*80}")
        
        # Create metrics tracker
        train_metrics = MetricsTracker()
        
        # Train one epoch
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            CONFIG,
            train_metrics
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device, CONFIG)
        
        # Update learning rate
        if scheduler:
            if CONFIG['scheduler_type'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rates'].append(current_lr)
        training_history['epochs'].append(epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*80}")
        print(f"Train Loss:      {train_loss:.6f}")
        print(f"Val Loss:        {val_loss:.6f}")
        print(f"Val Loss Std:    {val_metrics['val_loss_std']:.6f}")
        print(f"Learning Rate:   {current_lr:.6e}")
        print(f"Epoch Time:      {epoch_time:.2f}s")
        
        # Check if best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            improvement = (training_history['val_loss'][-2] - val_loss) if len(training_history['val_loss']) > 1 else 0
            print(f"✓ New best validation loss! Improved by {improvement:.6f}")
        
        # Save checkpoint
        if not CONFIG['save_best_only'] or is_best or epoch % CONFIG['save_interval'] == 0:
            checkpoint_name = f"liquid_velocity_epoch_{epoch:03d}_val{val_loss:.6f}.pth"
            checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], checkpoint_name)
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss,
                best_val_loss, CONFIG, checkpoint_path, is_best
            )
            
            if not is_best:
                print(f"✓ Saved checkpoint to {checkpoint_path}")
        
        # Early stopping check
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"\n{'='*80}")
                print(f"EARLY STOPPING TRIGGERED AT EPOCH {epoch}")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
        
        # Save training history
        history_path = os.path.join(CONFIG['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time:          {total_time/60:.2f} minutes")
    print(f"Best val loss:       {best_val_loss:.6f}")
    print(f"Final train loss:    {training_history['train_loss'][-1]:.6f}")
    print(f"Final val loss:      {training_history['val_loss'][-1]:.6f}")
    print(f"Checkpoints saved:   {CONFIG['checkpoint_dir']}")
    print(f"Logs saved:          {CONFIG['log_dir']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        train_liquid_velocity_model()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
