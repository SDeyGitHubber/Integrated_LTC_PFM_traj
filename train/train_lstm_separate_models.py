"""
Training Script for SEPARATE Linear and Non-Linear Models

This script:
1. Classifies trajectories as linear vs non-linear
2. Trains TWO SEPARATE models:
   - Model 1: Trained ONLY on linear trajectories
   - Model 2: Trained ONLY on non-linear trajectories
3. Saves both models separately with different checkpoint paths
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import os
from tqdm import tqdm
import json
import numpy as np

from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from models.lstm_ang_vel_unregularized import VelocityBasedLSTMUnregularized
from utils.trajectory_linearity_classifier import TrajectoryLinearityClassifier


def position_mse_loss(predictions, targets):
    """Direct MSE loss on positions."""
    return ((predictions - targets) ** 2).sum(dim=-1).mean()


def classify_entire_dataset(dataset, classifier, history_len, prediction_len, max_neighbors, device):
    """
    Pre-classify entire dataset into linear and non-linear indices.
    
    Returns:
        linear_indices: List of dataset indices for linear trajectories
        nonlinear_indices: List of dataset indices for non-linear trajectories
    """
    print("Pre-classifying entire dataset...")
    
    # Create temporary loader for classification
    temp_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, history_len, prediction_len, max_neighbors),
        num_workers=0
    )
    
    linear_indices = []
    nonlinear_indices = []
    
    current_idx = 0
    
    with torch.no_grad():
        for hist_neighbors, future, _, _, _, _ in tqdm(temp_loader, desc="Classifying"):
            hist_neighbors = hist_neighbors.to(device)
            future = future.to(device)
            
            history_ego = hist_neighbors[:, :, 0, :, :]
            classifications, _, _, _ = classifier.classify_batch(history_ego, future)
            
            B, A = classifications.shape
            
            # Extract indices
            for b in range(B):
                for a in range(A):
                    if classifications[b, a] == 1:
                        linear_indices.append(current_idx)
                    else:
                        nonlinear_indices.append(current_idx)
                    current_idx += 1
    
    print(f"Classification complete:")
    print(f"  Linear samples: {len(linear_indices)}")
    print(f"  Non-linear samples: {len(nonlinear_indices)}")
    print(f"  Linear ratio: {len(linear_indices) / (len(linear_indices) + len(nonlinear_indices)):.2%}\n")
    
    return linear_indices, nonlinear_indices


def train_one_epoch(model, train_loader, optimizer, device, config, epoch, model_name):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training {model_name}", ncols=120)
    
    for hist_neighbors, future, _, _, exp_goals, _ in pbar:
        hist_neighbors = hist_neighbors.to(device)
        future = future.to(device)
        
        optimizer.zero_grad()
        predictions, velocities, _ = model(hist_neighbors, exp_goals)
        
        loss = position_mse_loss(predictions, future)
        loss.backward()
        
        if config['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'avg': f"{total_loss/num_batches:.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, val_loader, device, model_name):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validating {model_name}", ncols=120)
        
        for hist_neighbors, future, _, _, exp_goals, _ in pbar:
            hist_neighbors = hist_neighbors.to(device)
            future = future.to(device)
            
            predictions, velocities, _ = model(hist_neighbors, exp_goals)
            loss = position_mse_loss(predictions, future)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'avg': f"{total_loss/num_batches:.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train_separate_models():
    """
    Main training function - trains TWO SEPARATE models.
    """
    
    CONFIG = {
        # Data
        'data_path': 'data/combined_annotations.csv',
        'checkpoint_dir_linear': 'checkpoints/lstm_unregularized_LINEAR',
        'checkpoint_dir_nonlinear': 'checkpoints/lstm_unregularized_NONLINEAR',
        'log_dir': 'logs/lstm_separate_models',
        'train_split': 0.8,
        'random_seed': 42,
        
        # Model
        'hidden_size': 64,
        'num_layers': 2,
        'dt': 0.25,
        'output_mode': 'velocities',
        'use_speed_limits': False,
        'max_speed': 10.0,
        
        # Dataset
        'history_len': 8,
        'prediction_len': 12,
        'max_neighbors': 4,
        
        # Training
        'batch_size': 16,
        'learning_rate': 1e-3,
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
        
        # Checkpointing
        'save_interval': 5,
    }
    
    # Setup directories
    os.makedirs(CONFIG['checkpoint_dir_linear'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir_nonlinear'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("TRAINING TWO SEPARATE MODELS (LINEAR vs NON-LINEAR)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Linear checkpoints: {CONFIG['checkpoint_dir_linear']}")
    print(f"Non-linear checkpoints: {CONFIG['checkpoint_dir_nonlinear']}")
    print("="*80 + "\n")
    
    # Load full dataset
    print("Loading dataset...")
    
    # Try multiple possible paths
    possible_paths = [
        CONFIG['data_path'],
        'data/combined_annotations.csv',
        '../data/combined_annotations.csv',
        '/kaggle/input/combined_annotations.csv',
        '/content/combined_annotations.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            print(f"✓ Found dataset at: {path}")
            break
    
    if data_path is None:
        raise FileNotFoundError(
            f"Dataset not found! Tried the following paths:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths) + 
            "\n\nPlease update CONFIG['data_path'] to your actual file location."
        )
    
    full_dataset = PFM_TrajectoryDataset_neighbours(
        data_path,
        history_len=CONFIG['history_len'],
        prediction_len=CONFIG['prediction_len'],
        max_neighbors=CONFIG['max_neighbors']
    )
    
    if len(full_dataset) == 0:
        raise ValueError(
            f"Dataset is empty! File: {data_path}\n"
            "Please check:\n"
            "1. The file contains valid data\n"
            "2. The file path is correct\n"
            "3. The data format matches the expected structure"
        )
    
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty!")
    
    # Initialize classifier
    classifier = TrajectoryLinearityClassifier(
        threshold_method=CONFIG['linearity_threshold_method'],
        k_std=CONFIG['linearity_k_std']
    )
    
    # Pre-classify entire dataset
    linear_indices, nonlinear_indices = classify_entire_dataset(
        full_dataset, classifier, 
        CONFIG['history_len'], CONFIG['prediction_len'], 
        CONFIG['max_neighbors'], device
    )
    
    # Create separate datasets for linear and non-linear
    linear_dataset = Subset(full_dataset, linear_indices)
    nonlinear_dataset = Subset(full_dataset, nonlinear_indices)
    
    print(f"Linear dataset size: {len(linear_dataset)}")
    print(f"Non-linear dataset size: {len(nonlinear_dataset)}\n")
    
    # Split into train/val for LINEAR
    linear_total = len(linear_dataset)
    linear_train_size = int(CONFIG['train_split'] * linear_total)
    linear_val_size = linear_total - linear_train_size
    
    linear_train_dataset, linear_val_dataset = random_split(
        linear_dataset,
        [linear_train_size, linear_val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    
    # Split into train/val for NON-LINEAR
    nonlinear_total = len(nonlinear_dataset)
    nonlinear_train_size = int(CONFIG['train_split'] * nonlinear_total)
    nonlinear_val_size = nonlinear_total - nonlinear_train_size
    
    nonlinear_train_dataset, nonlinear_val_dataset = random_split(
        nonlinear_dataset,
        [nonlinear_train_size, nonlinear_val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    
    print(f"LINEAR - Train: {linear_train_size}, Val: {linear_val_size}")
    print(f"NON-LINEAR - Train: {nonlinear_train_size}, Val: {nonlinear_val_size}\n")
    
    # Create data loaders for LINEAR
    linear_train_loader = DataLoader(
        linear_train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, CONFIG['history_len'], CONFIG['prediction_len'], CONFIG['max_neighbors']),
        num_workers=0
    )
    
    linear_val_loader = DataLoader(
        linear_val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, CONFIG['history_len'], CONFIG['prediction_len'], CONFIG['max_neighbors']),
        num_workers=0
    )
    
    # Create data loaders for NON-LINEAR
    nonlinear_train_loader = DataLoader(
        nonlinear_train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, CONFIG['history_len'], CONFIG['prediction_len'], CONFIG['max_neighbors']),
        num_workers=0
    )
    
    nonlinear_val_loader = DataLoader(
        nonlinear_val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, CONFIG['history_len'], CONFIG['prediction_len'], CONFIG['max_neighbors']),
        num_workers=0
    )
    
    # ========================================================================
    # Initialize TWO SEPARATE MODELS
    # ========================================================================
    
    print("="*80)
    print("INITIALIZING LINEAR MODEL")
    print("="*80)
    
    linear_model = VelocityBasedLSTMUnregularized(
        input_size=2,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dt=CONFIG['dt'],
        output_mode=CONFIG['output_mode'],
        use_speed_limits=CONFIG['use_speed_limits'],
        max_speed=CONFIG['max_speed']
    )
    linear_model.to(device)
    print(f"Parameters: {sum(p.numel() for p in linear_model.parameters()):,}\n")
    
    print("="*80)
    print("INITIALIZING NON-LINEAR MODEL")
    print("="*80)
    
    nonlinear_model = VelocityBasedLSTMUnregularized(
        input_size=2,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dt=CONFIG['dt'],
        output_mode=CONFIG['output_mode'],
        use_speed_limits=CONFIG['use_speed_limits'],
        max_speed=CONFIG['max_speed']
    )
    nonlinear_model.to(device)
    print(f"Parameters: {sum(p.numel() for p in nonlinear_model.parameters()):,}\n")
    
    # Optimizers
    linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=CONFIG['learning_rate'])
    nonlinear_optimizer = torch.optim.Adam(nonlinear_model.parameters(), lr=CONFIG['learning_rate'])
    
    # Schedulers
    linear_scheduler = None
    nonlinear_scheduler = None
    
    if CONFIG['use_scheduler']:
        linear_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            linear_optimizer,
            mode='min',
            factor=CONFIG['scheduler_factor'],
            patience=CONFIG['scheduler_patience'],
            min_lr=CONFIG['scheduler_min_lr'],
            verbose=True
        )
        
        nonlinear_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            nonlinear_optimizer,
            mode='min',
            factor=CONFIG['scheduler_factor'],
            patience=CONFIG['scheduler_patience'],
            min_lr=CONFIG['scheduler_min_lr'],
            verbose=True
        )
    
    # Training history
    history = {
        'linear_train_loss': [],
        'linear_val_loss': [],
        'nonlinear_train_loss': [],
        'nonlinear_val_loss': [],
        'epochs': []
    }
    
    best_linear_loss = float('inf')
    best_nonlinear_loss = float('inf')
    
    # ========================================================================
    # TRAINING LOOP - TRAIN BOTH MODELS SEPARATELY
    # ========================================================================
    
    print("="*80)
    print("STARTING TRAINING (TWO SEPARATE MODELS)")
    print("="*80 + "\n")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{CONFIG['num_epochs']}")
        print(f"{'='*80}\n")
        
        # ====================================================================
        # Train LINEAR model
        # ====================================================================
        print("→ Training LINEAR model...")
        linear_train_loss = train_one_epoch(
            linear_model, linear_train_loader, linear_optimizer, 
            device, CONFIG, epoch, "LINEAR"
        )
        
        linear_val_loss = validate(linear_model, linear_val_loader, device, "LINEAR")
        
        print(f"LINEAR - Train Loss: {linear_train_loss:.6f} | Val Loss: {linear_val_loss:.6f}")
        
        # Update LINEAR scheduler
        if linear_scheduler:
            linear_scheduler.step(linear_val_loss)
        
        # ====================================================================
        # Train NON-LINEAR model
        # ====================================================================
        print("\n→ Training NON-LINEAR model...")
        nonlinear_train_loss = train_one_epoch(
            nonlinear_model, nonlinear_train_loader, nonlinear_optimizer, 
            device, CONFIG, epoch, "NON-LINEAR"
        )
        
        nonlinear_val_loss = validate(nonlinear_model, nonlinear_val_loader, device, "NON-LINEAR")
        
        print(f"NON-LINEAR - Train Loss: {nonlinear_train_loss:.6f} | Val Loss: {nonlinear_val_loss:.6f}")
        
        # Update NON-LINEAR scheduler
        if nonlinear_scheduler:
            nonlinear_scheduler.step(nonlinear_val_loss)
        
        # ====================================================================
        # Track history
        # ====================================================================
        history['linear_train_loss'].append(linear_train_loss)
        history['linear_val_loss'].append(linear_val_loss)
        history['nonlinear_train_loss'].append(nonlinear_train_loss)
        history['nonlinear_val_loss'].append(nonlinear_val_loss)
        history['epochs'].append(epoch)
        
        # ====================================================================
        # Save checkpoints for BOTH models
        # ====================================================================
        
        # Save LINEAR model
        is_best_linear = linear_val_loss < best_linear_loss
        if is_best_linear:
            best_linear_loss = linear_val_loss
        
        if epoch % CONFIG['save_interval'] == 0 or is_best_linear:
            linear_checkpoint = {
                'epoch': epoch,
                'model_state_dict': linear_model.state_dict(),
                'optimizer_state_dict': linear_optimizer.state_dict(),
                'config': CONFIG,
                'val_loss': linear_val_loss,
                'best_val_loss': best_linear_loss,
                'model_type': 'LINEAR'
            }
            
            linear_save_path = os.path.join(CONFIG['checkpoint_dir_linear'], f"epoch_{epoch:03d}.pth")
            torch.save(linear_checkpoint, linear_save_path)
            
            if is_best_linear:
                linear_best_path = os.path.join(CONFIG['checkpoint_dir_linear'], "best_linear_model.pth")
                torch.save(linear_checkpoint, linear_best_path)
                print(f"✓ Saved BEST LINEAR model: {linear_best_path}")
        
        # Save NON-LINEAR model
        is_best_nonlinear = nonlinear_val_loss < best_nonlinear_loss
        if is_best_nonlinear:
            best_nonlinear_loss = nonlinear_val_loss
        
        if epoch % CONFIG['save_interval'] == 0 or is_best_nonlinear:
            nonlinear_checkpoint = {
                'epoch': epoch,
                'model_state_dict': nonlinear_model.state_dict(),
                'optimizer_state_dict': nonlinear_optimizer.state_dict(),
                'config': CONFIG,
                'val_loss': nonlinear_val_loss,
                'best_val_loss': best_nonlinear_loss,
                'model_type': 'NON-LINEAR'
            }
            
            nonlinear_save_path = os.path.join(CONFIG['checkpoint_dir_nonlinear'], f"epoch_{epoch:03d}.pth")
            torch.save(nonlinear_checkpoint, nonlinear_save_path)
            
            if is_best_nonlinear:
                nonlinear_best_path = os.path.join(CONFIG['checkpoint_dir_nonlinear'], "best_nonlinear_model.pth")
                torch.save(nonlinear_checkpoint, nonlinear_best_path)
                print(f"✓ Saved BEST NON-LINEAR model: {nonlinear_best_path}")
        
        # Save training history
        with open(os.path.join(CONFIG['log_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*80}")
        print(f"LINEAR     → Train: {linear_train_loss:.6f} | Val: {linear_val_loss:.6f} | Best: {best_linear_loss:.6f}")
        print(f"NON-LINEAR → Train: {nonlinear_train_loss:.6f} | Val: {nonlinear_val_loss:.6f} | Best: {best_nonlinear_loss:.6f}")
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Best LINEAR model loss: {best_linear_loss:.6f}")
    print(f"  Best NON-LINEAR model loss: {best_nonlinear_loss:.6f}")
    print(f"\nModel Paths:")
    print(f"  LINEAR: {CONFIG['checkpoint_dir_linear']}/best_linear_model.pth")
    print(f"  NON-LINEAR: {CONFIG['checkpoint_dir_nonlinear']}/best_nonlinear_model.pth")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_separate_models()
