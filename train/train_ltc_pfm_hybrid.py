"""
================================================================================
Training Script for LTC-PFM HYBRID MODEL
================================================================================

Trains the hybrid Liquid Time-Constant Network with Physics-Informed Motion
for multi-agent trajectory prediction.

Key Features:
------------
1. CfC/LTC Encoder-Decoder: Continuous-time neural dynamics
2. Potential Field Module (PFM): Physics-based motion priors
3. Neighbor Interaction: Multi-agent trajectory prediction
4. Advanced Training: Gradient checkpointing, LR scheduling, early stopping
5. Tweakable ODE Parameters: Full access to CfC internals

Architecture:
-------------
1. Input: history_neighbors [B, A, ent, H, 2] + goals [B, A, ent, 2]
2. CfC Encoder: Encodes history with tweakable continuous-time dynamics
3. PFM Module: Computes physics-based forces (goal, prediction, repulsion)
4. CfC Decoder: Autoregressive decoding with PFM conditioning
5. Output: adjusted_preds (physics-corrected) + decoded_preds (raw neural)

Loss Function:
--------------
Hybrid Loss: Combines position MSE with coefficient regularization
    L = MSE(adjusted_preds, targets) + λ·regularization(coefficients)
    
This encourages:
- Accurate position predictions
- Stable, interpretable PFM coefficients
- Physical plausibility

References:
----------
[1] Hasani et al., "Liquid Time-Constant Networks", AAAI 2021
[2] Hasani et al., "Closed-form Continuous-time Neural Networks", Nature ML 2022
[3] Current work: LTC-PFM Hybrid Model for trajectory prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import json
import time
from datetime import datetime
import numpy as np
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from models.ltc_pfm_hybrid_model import LTC_PFM_HybridModel


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Monitors validation loss and stops training if no improvement
    is observed for a specified number of epochs (patience).
    """
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
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
                print(f"  [EarlyStopping] No improvement: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] Early stopping triggered!")
        else:
            improvement = self.best_loss - val_loss
            if self.verbose:
                print(f"  [EarlyStopping] Validation improved by {improvement:.6f}")
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
        self.total_position_loss = 0.0
        self.total_coeff_reg = 0.0
        self.num_batches = 0
        self.batch_losses = []
    
    def update(self, loss, position_loss=0.0, coeff_reg=0.0):
        """Update metrics with batch results."""
        self.total_loss += loss
        self.total_position_loss += position_loss
        self.total_coeff_reg += coeff_reg
        self.num_batches += 1
        self.batch_losses.append(loss)
    
    def get_average_loss(self):
        """Get average loss across all batches."""
        return self.total_loss / max(self.num_batches, 1)
    
    def get_average_position_loss(self):
        """Get average position loss."""
        return self.total_position_loss / max(self.num_batches, 1)
    
    def get_average_coeff_reg(self):
        """Get average coefficient regularization."""
        return self.total_coeff_reg / max(self.num_batches, 1)
    
    def get_loss_std(self):
        """Get standard deviation of batch losses."""
        if len(self.batch_losses) < 2:
            return 0.0
        return np.std(self.batch_losses)


def compute_ade(predictions, targets):
    """
    Compute Average Displacement Error (ADE).
    
    ADE = (1/T) * Σ ||pred_t - target_t||₂
    
    Args:
        predictions: [B, A, ent, T, 2] predicted positions
        targets: [B, A, ent, T, 2] ground truth positions
    
    Returns:
        ade: Average displacement error (scalar)
    """
    displacements = torch.norm(predictions - targets, dim=-1)  # [B, A, ent, T]
    ade = displacements.mean()
    return ade


def compute_fde(predictions, targets):
    """
    Compute Final Displacement Error (FDE).
    
    FDE = ||pred_T - target_T||₂
    
    Args:
        predictions: [B, A, ent, T, 2] predicted positions
        targets: [B, A, ent, T, 2] ground truth positions
    
    Returns:
        fde: Final displacement error (scalar)
    """
    final_displacements = torch.norm(
        predictions[..., -1, :] - targets[..., -1, :],
        dim=-1
    )  # [B, A, ent]
    fde = final_displacements.mean()
    return fde


def position_loss(adjusted_preds, decoded_preds, targets, config):
    """
    Compute hybrid position loss.
    
    Combines:
    1. MSE on physics-corrected predictions (adjusted_preds)
    2. MSE on raw neural predictions (decoded_preds) 
    3. Optional coefficient regularization
    
    Args:
        adjusted_preds: [B, A, ent, T, 2] physics-corrected predictions
        decoded_preds: [B, A, ent, T, 2] raw neural predictions
        targets: [B, A, ent, T, 2] ground truth
        config: Configuration dictionary
    
    Returns:
        loss: Total weighted loss (scalar)
        loss_dict: Dictionary of loss components
    """
    # Only use ego agent (index 0) for loss computation
    adjusted_ego = adjusted_preds[:, :, 0]  # [B, A, T, 2]
    decoded_ego = decoded_preds[:, :, 0]    # [B, A, T, 2]
    targets_ego = targets[:, :, 0]          # [B, A, T, 2]
    
    # Position loss: Adjusted predictions (physics-corrected)
    adjusted_loss = F.mse_loss(adjusted_ego, targets_ego)
    
    # Auxiliary loss: Raw neural predictions
    decoded_loss = F.mse_loss(decoded_ego, targets_ego)
    
    # Total loss
    alpha = config.get('loss_alpha', 0.8)  # Weight for adjusted vs decoded
    loss = alpha * adjusted_loss + (1 - alpha) * decoded_loss
    
    loss_dict = {
        'adjusted_loss': adjusted_loss.item(),
        'decoded_loss': decoded_loss.item(),
        'total_loss': loss.item()
    }
    
    return loss, loss_dict


def train_one_epoch(model, train_loader, optimizer, device, config, metrics_tracker):
    """
    Train model for one epoch.
    
    Args:
        model: LTC_PFM_HybridModel instance
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
    
    pbar = tqdm(train_loader, desc="Training", ncols=140)
    
    for batch_idx, (hist_neighbors, future, _, _, exp_goals, all_futures) in enumerate(pbar):
        # Move data to device
        hist_neighbors = hist_neighbors.to(device)
        exp_goals = exp_goals.to(device)
        all_futures = all_futures.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        adjusted_preds, decoded_preds, coeff_mean, coeff_var = model(
            hist_neighbors, exp_goals
        )
        
        # Compute loss
        loss, loss_dict = position_loss(adjusted_preds, decoded_preds, all_futures, config)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (critical for CfC stability)
        if config['gradient_clip'] > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['gradient_clip']
            )
        else:
            grad_norm = 0.0
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        metrics_tracker.update(
            loss.item(),
            position_loss=loss_dict['adjusted_loss'],
            coeff_reg=coeff_var.item()
        )
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.6f}",
            'adj_loss': f"{loss_dict['adjusted_loss']:.6f}",
            'dec_loss': f"{loss_dict['decoded_loss']:.6f}",
            'coeff_var': f"{coeff_var.item():.6e}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6e}",
            'grad_norm': f"{grad_norm:.3f}" if grad_norm > 0 else "N/A"
        })
    
    return metrics_tracker.get_average_loss()


def validate(model, val_loader, device, config):
    """
    Validate model on validation set.
    
    Args:
        model: LTC_PFM_HybridModel instance
        val_loader: DataLoader for validation data
        device: torch.device
        config: Configuration dictionary
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary of validation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    ade_tracker = []
    fde_tracker = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", ncols=140)
        
        for hist_neighbors, future, _, _, exp_goals, all_futures in pbar:
            # Move data to device
            hist_neighbors = hist_neighbors.to(device)
            exp_goals = exp_goals.to(device)
            all_futures = all_futures.to(device)
            
            # Forward pass
            adjusted_preds, decoded_preds, coeff_mean, coeff_var = model(
                hist_neighbors, exp_goals
            )
            
            # Compute loss
            loss, loss_dict = position_loss(adjusted_preds, decoded_preds, all_futures, config)
            
            # Compute metrics
            ade = compute_ade(adjusted_preds, all_futures)
            fde = compute_fde(adjusted_preds, all_futures)
            
            ade_tracker.append(ade.item())
            fde_tracker.append(fde.item())
            
            # Update metrics
            metrics_tracker.update(
                loss.item(),
                position_loss=loss_dict['adjusted_loss'],
                coeff_reg=coeff_var.item()
            )
            
            # Update progress bar
            pbar.set_postfix({
                'val_loss': f"{loss.item():.6f}",
                'ade': f"{ade.item():.6f}",
                'fde': f"{fde.item():.6f}",
                'coeff_var': f"{coeff_var.item():.6e}"
            })
    
    avg_loss = metrics_tracker.get_average_loss()
    loss_std = metrics_tracker.get_loss_std()
    avg_ade = np.mean(ade_tracker)
    avg_fde = np.mean(fde_tracker)
    
    metrics = {
        'val_loss': avg_loss,
        'val_loss_std': loss_std,
        'val_ade': avg_ade,
        'val_fde': avg_fde,
        'num_batches': metrics_tracker.num_batches
    }
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss,
                   best_val_loss, config, filepath, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: LTC_PFM_HybridModel instance
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
            "best_ltc_pfm_hybrid_model.pth"
        )
        torch.save(checkpoint, best_path)
        print(f"  ✓ Saved best model to {best_path}")


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
    
    print("\n" + "="*100)
    print("LTC-PFM HYBRID MODEL TRAINING CONFIGURATION")
    print("="*100)
    for key, value in config.items():
        print(f"{key:35s}: {value}")
    print("="*100 + "\n")


def train_ltc_pfm_hybrid_model():
    """
    Main training function for LTC-PFM Hybrid Model.
    
    Trains a Liquid Time-Constant Network with Physics-Informed Potential Fields
    for continuous-time trajectory prediction with social context.
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    CONFIG = {
        # Data paths
        'data_path': 'data/combined_annotations.csv',
        'checkpoint_dir': 'checkpoints/ltc_pfm_hybrid',
        'log_dir': 'logs/ltc_pfm_hybrid',
        
        # Data split
        'train_split': 0.8,
        'random_seed': 42,
        
        # ====================================================================
        # CfC ENCODER/DECODER PARAMETERS (TWEAKABLE)
        # ====================================================================
        'hidden_size': 64,              # CfC hidden units
        'cfc_mode': 'default',          # CfC mode: "default", "pure", "no_gate"
        'cfc_backbone_units': 128,      # Backbone MLP units
        'cfc_backbone_layers': 1,       # Backbone MLP layers
        'cfc_backbone_activation': 'lecun_tanh',  # Activation: lecun_tanh, relu, gelu, etc.
        'cfc_backbone_dropout': 0.1,    # Dropout in backbone
        'mixed_memory': True,           # Augment with LSTM cell
        
        # ====================================================================
        # MODEL PARAMETERS
        # ====================================================================
        'dt': 0.1,                      # Integration timestep (seconds)
        'target_avg_speed': 4.087,      # Average speed
        'speed_tolerance': 0.15,        # Speed clamping tolerance
        'use_angular_velocity': True,   # Enable (Δv, Δω) mode
        'num_agents': 1000,             # Max agents for embedding
        
        # ====================================================================
        # DATASET PARAMETERS
        # ====================================================================
        'history_len': 8,               # History length (timesteps)
        'prediction_len': 12,           # Prediction length (timesteps)
        'max_neighbors': 12,            # Max neighbors per agent
        
        # ====================================================================
        # TRAINING HYPERPARAMETERS
        # ====================================================================
        'batch_size': 8,                # Batch size (reduce if CUDA OOM)
        'learning_rate': 1e-4,          # Initial learning rate (CfC is sensitive)
        'num_epochs': 100,              # Max epochs
        'gradient_clip': 1.0,           # Gradient clipping threshold
        'loss_alpha': 0.8,              # Weight for adjusted vs decoded predictions
        
        # ====================================================================
        # LEARNING RATE SCHEDULING
        # ====================================================================
        'use_scheduler': True,
        'scheduler_type': 'plateau',    # 'plateau' or 'cosine'
        'scheduler_patience': 7,        # ReduceLROnPlateau patience
        'scheduler_factor': 0.5,        # LR reduction factor
        'scheduler_min_lr': 1e-7,       # Minimum learning rate
        'scheduler_warmup_epochs': 0,   # Warmup epochs (0 = no warmup)
        
        # ====================================================================
        # EARLY STOPPING
        # ====================================================================
        'use_early_stopping': True,
        'early_stopping_patience': 15,
        'early_stopping_min_delta': 1e-4,
        
        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        'save_best_only': False,        # Save all checkpoints or best only
        'save_interval': 5,             # Save checkpoint every N epochs
        
        # ====================================================================
        # DATALOADER
        # ====================================================================
        'num_workers': 2,               # Data loading workers
        'pin_memory': True,             # Pin memory for faster GPU transfer
        'persistent_workers': False,    # Keep workers alive between epochs
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
    possible_paths = [
        CONFIG['data_path'],
        '/content/combined_annotations.csv',
        'combined_annotations.csv',
        'data/crowds_zara02_test_cleaned.txt',
        '../data/combined_annotations.csv',
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
        raise FileNotFoundError(error_msg)
    
    CONFIG['data_path'] = data_path_found
    
    # Log configuration
    log_training_info(CONFIG, CONFIG['log_dir'])
    
    # Set random seed
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['random_seed'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'Device':<40}: {device}")
    if torch.cuda.is_available():
        print(f"{'GPU':<40}: {torch.cuda.get_device_name(0)}")
        print(f"{'GPU Memory':<40}: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    print("="*100)
    print("LOADING DATASET")
    print("="*100)
    
    # Load dataset
    dataset = PFM_TrajectoryDataset_neighbours(
        file_path=CONFIG['data_path'],
        history_len=CONFIG['history_len'],
        prediction_len=CONFIG['prediction_len'],
        max_neighbors=CONFIG['max_neighbors']
    )
    
    print(f"Total samples: {len(dataset)}")
    
    # Split into train/validation
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=CONFIG['persistent_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=CONFIG['persistent_workers']
    )
    
    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    
    print("\n" + "="*100)
    print("INITIALIZING MODEL")
    print("="*100)
    
    model = LTC_PFM_HybridModel(
        input_size=2,
        hidden_size=CONFIG['hidden_size'],
        num_layers=1,
        target_avg_speed=CONFIG['target_avg_speed'],
        speed_tolerance=CONFIG['speed_tolerance'],
        num_agents=CONFIG['num_agents'],
        dt=CONFIG['dt'],
        # CfC parameters (TWEAKABLE)
        cfc_mode=CONFIG['cfc_mode'],
        cfc_backbone_units=CONFIG['cfc_backbone_units'],
        cfc_backbone_layers=CONFIG['cfc_backbone_layers'],
        cfc_backbone_activation=CONFIG['cfc_backbone_activation'],
        cfc_backbone_dropout=CONFIG['cfc_backbone_dropout'],
        mixed_memory=CONFIG['mixed_memory'],
        use_angular_velocity=CONFIG['use_angular_velocity']
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'Model':<40}: LTC_PFM_HybridModel")
    print(f"{'Total parameters':<40}: {total_params:,}")
    print(f"{'Trainable parameters':<40}: {trainable_params:,}")
    print(f"{'CfC Mode':<40}: {CONFIG['cfc_mode']}")
    print(f"{'Hidden Size':<40}: {CONFIG['hidden_size']}")
    print(f"{'Backbone Activation':<40}: {CONFIG['cfc_backbone_activation']}")
    
    # ========================================================================
    # OPTIMIZER AND SCHEDULER
    # ========================================================================
    
    print("\n" + "="*100)
    print("OPTIMIZER CONFIGURATION")
    print("="*100)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5
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
    
    print(f"{'Optimizer':<40}: Adam")
    print(f"{'Learning Rate':<40}: {CONFIG['learning_rate']:.6e}")
    print(f"{'Gradient Clipping':<40}: {CONFIG['gradient_clip']}")
    print(f"{'Use Scheduler':<40}: {CONFIG['use_scheduler']}")
    if CONFIG['use_scheduler']:
        print(f"{'Scheduler Type':<40}: {CONFIG['scheduler_type']}")
    
    # ========================================================================
    # EARLY STOPPING
    # ========================================================================
    
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
    
    print("\n" + "="*100)
    print("TRAINING")
    print("="*100 + "\n")
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_ade': [],
        'val_fde': [],
        'learning_rate': []
    }
    
    for epoch in range(CONFIG['num_epochs']):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 100)
        
        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, CONFIG, MetricsTracker()
        )
        
        # Validation
        val_loss, val_metrics = validate(model, val_loader, device, CONFIG)
        
        epoch_time = time.time() - epoch_start
        
        # Learning rate scheduler
        if scheduler is not None:
            if CONFIG['scheduler_type'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("\n[Training] Early stopping triggered!")
                break
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % CONFIG['save_interval'] == 0 or is_best:
            checkpoint_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"model_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, train_loss, val_loss,
                best_val_loss, CONFIG, checkpoint_path, is_best=is_best
            )
        
        # Log metrics
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_ade'].append(val_metrics['val_ade'])
        training_history['val_fde'].append(val_metrics['val_fde'])
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print summary
        print(f"\n  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f} (best: {best_val_loss:.6f})")
        print(f"  Val ADE:    {val_metrics['val_ade']:.6f}")
        print(f"  Val FDE:    {val_metrics['val_fde']:.6f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6e}")
        print(f"  Time:       {epoch_time:.1f}s")
    
    # ========================================================================
    # TRAINING COMPLETED
    # ========================================================================
    
    total_time = time.time() - start_time
    
    print("\n" + "="*100)
    print("TRAINING COMPLETED")
    print("="*100)
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint directory: {CONFIG['checkpoint_dir']}")
    
    # Save training history
    history_path = os.path.join(CONFIG['log_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    
    print(f"Training history saved to: {history_path}")
    
    return model, CONFIG, training_history


if __name__ == "__main__":
    model, config, history = train_ltc_pfm_hybrid_model()
    print("\n✓ Training complete!")
