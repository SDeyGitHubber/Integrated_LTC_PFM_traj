# train_lstm_velocity_full.py
"""
Production Training Script for Velocity-Based LSTM Model
- Full dataset (no subsampling)
- Train/Validation split (80/20)
- Early stopping to prevent overfitting
- 20 epochs with checkpointing
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import json
from datetime import datetime

from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from models.mta_lstm_ang_vel import VelocityBasedLSTM


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0.0, verbose=True):
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
                print(f"[EarlyStopping] Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def residual_velocity_loss(pred_vel_residuals):
    return pred_vel_residuals.norm(dim = -1).mean()

def train_one_epoch(model, train_loader, optimizer, device, gradient_clip):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for hist_neighbors, _, _, _, exp_goals, _ in pbar:
        hist_neighbors = hist_neighbors.to(device)
        # future = future.to(device)   #ignore the future argument for residual training
        optimizer.zero_grad()
        predictions, velocity_residuals, _ = model(hist_neighbors, exp_goals)
        loss = residual_velocity_loss(velocity_residuals)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.6f}"})

    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for hist_neighbors, _, _, _, exp_goals, _ in pbar:
            hist_neighbors = hist_neighbors.to(device)

            predictions, velocity_residuals, _ = model(hist_neighbors, exp_goals)
            loss = residual_velocity_loss(velocity_residuals)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'val_loss': f"{loss.item():.6f}"})

    return total_loss / num_batches


def train_velocity_lstm_full():
    """
    Full training with validation and early stopping.
    """
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    CONFIG = {
        'data_path': './data/combined_annotations.csv',
        'checkpoint_dir': './checkpoints/velocity_lstm_full',
        'log_dir': 'logs_velocity_lstm_full',

        # Dataset split
        'train_split': 0.8,  # 80% train, 20% validation

        # Model hyperparameters
        'hidden_size': 64,
        'num_layers': 2,
        'dt': 0.25,
        'target_avg_speed': 20.114,  # From dataset calculation
        'speed_tolerance': 0.15,

        # Training hyperparameters
        'batch_size': 8,
        'learning_rate': 1e-3,
        'num_epochs': 20,
        'gradient_clip': 1.0,

        # Early stopping
        'early_stopping_patience': 7,
        'early_stopping_min_delta': 0.001,

        # Scheduler
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,

        # Logging
        'log_interval': 10,
        'save_best_only': False,  # Save all checkpoints
    }

    # Create directories
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"VELOCITY-BASED LSTM - FULL TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Training: {CONFIG['train_split']*100:.0f}% | Validation: {(1-CONFIG['train_split'])*100:.0f}%")

    # =====================================================================
    # LOAD DATASET
    # =====================================================================
    print(f"\n[1/7] Loading full dataset...")
    full_dataset = PFM_TrajectoryDataset_neighbours(
        CONFIG['data_path'],
        history_len=8,
        prediction_len=12,
        max_neighbors=12
    )

    total_samples = len(full_dataset)
    train_size = int(CONFIG['train_split'] * total_samples)
    val_size = total_samples - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Total samples: {total_samples}")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")

    # =====================================================================
    # CREATE DATALOADERS
    # =====================================================================
    print(f"\n[2/7] Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, 8, 12, 12),
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, 8, 12, 12),
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # =====================================================================
    # INITIALIZE MODEL
    # =====================================================================
    print(f"\n[3/7] Initializing model...")
    model = VelocityBasedLSTM(
        input_size=2,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dt=CONFIG['dt'],
        target_avg_speed=CONFIG['target_avg_speed'],
        speed_tolerance=CONFIG['speed_tolerance']
    )
    model.to(device)

    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"  Total parameters: {total_params:,}")
    # print(f"  Trainable parameters: {trainable_params:,}")

    # =====================================================================
    # SETUP TRAINING
    # =====================================================================
    print(f"\n[4/7] Setting up training components...")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG['scheduler_factor'],
        patience=CONFIG['scheduler_patience'],
        verbose=True
    )
    early_stopping = EarlyStopping(
        patience=CONFIG['early_stopping_patience'],
        min_delta=CONFIG['early_stopping_min_delta']
    )

    print(f"  Loss: MSELoss")
    print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']})")
    print(f"  Scheduler: ReduceLROnPlateau (patience={CONFIG['scheduler_patience']})")
    print(f"  Early Stopping: patience={CONFIG['early_stopping_patience']}")

    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    print(f"\n[5/7] Starting training...")
    print(f"{'='*70}")

    best_val_loss = float('inf')

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{CONFIG['num_epochs']}")
        print(f"{'='*70}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, CONFIG['gradient_clip']
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': CONFIG,
        }

        if not CONFIG['save_best_only'] or is_best:
            # Save epoch checkpoint
            epoch_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"epoch_{epoch:02d}_val{val_loss:.6f}.pth"
            )
            torch.save(checkpoint, epoch_path)
            print(f"\n[SAVED] {epoch_path}")

            # Save best
            if is_best:
                best_path = os.path.join(CONFIG['checkpoint_dir'], "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f"[SAVED] Best model: {best_path}")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n{'='*70}")
            print(f"⚠️  EARLY STOPPING TRIGGERED at Epoch {epoch}")
            print(f"{'='*70}")
            break

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    # Train
    train_velocity_lstm_full()