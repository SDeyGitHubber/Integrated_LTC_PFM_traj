"""
================================================================================
Training Script — TrajectronLTC
  Trajectron++ Encoder  →  CfC/LTC Decoder  →  PFM Adjustment
================================================================================

Usage (from workspace root):
    python train/train_trajectron_ltc.py
    python train/train_trajectron_ltc.py --epochs 100 --batch_size 8
    python train/train_trajectron_ltc.py --resume checkpoints/trajectron_ltc_best.pth

Key output per epoch:
  ┌ Epoch header with LR + KL-weight
  ├ Progress bars (train / val)
  ├ Loss breakdown table  (total | NLL-adj | NLL-dec | KL | entity)
  ├ ADE / FDE metrics
  ├ PFM coefficient stats
  └ Best-model marker & checkpoint paths
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add parent directory
try:
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    _BASE_DIR = os.path.abspath('..')

if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from models.trajectron_ltc_model import TrajectronLTC, ModeKeys


# =============================================================================
# EARLY STOPPING
# =============================================================================
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
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
                print(f"  [EarlyStopping] No improvement: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"  [EarlyStopping] Improved by {self.best_loss - val_loss:.6f}")
            self.best_loss = val_loss
            self.counter = 0


# =============================================================================
# METRICS TRACKER
# =============================================================================
class MetricsTracker:
    """Tracks and computes training metrics."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.nll_adjusted = []
        self.nll_decoded = []
        self.kl_losses = []
        self.entity_losses = []
        self.coeff_means = []
        self.coeff_vars = []
    
    def update(self, loss_dict):
        self.losses.append(loss_dict['total_loss'])
        self.nll_adjusted.append(loss_dict['nll_adjusted'])
        self.nll_decoded.append(loss_dict['nll_decoded'])
        self.kl_losses.append(loss_dict['kl_loss'])
        self.entity_losses.append(loss_dict['entity_loss'])
        self.coeff_means.append(loss_dict['coeff_mean'])
        self.coeff_vars.append(loss_dict['coeff_var'])
    
    def summary(self):
        return {
            'total_loss': np.mean(self.losses),
            'nll_adjusted': np.mean(self.nll_adjusted),
            'nll_decoded': np.mean(self.nll_decoded),
            'kl_loss': np.mean(self.kl_losses),
            'entity_loss': np.mean(self.entity_losses),
            'coeff_mean': np.mean(self.coeff_means),
            'coeff_var': np.mean(self.coeff_vars),
        }


# =============================================================================
# ADE / FDE METRICS
# =============================================================================
def compute_ade(pred, gt):
    """Average Displacement Error."""
    return torch.norm(pred - gt, dim=-1).mean().item()


def compute_fde(pred, gt):
    """Final Displacement Error."""
    return torch.norm(pred[:, :, -1, :] - gt[:, :, -1, :], dim=-1).mean().item()


# =============================================================================
# KL ANNEALING SCHEDULER
# =============================================================================
class KLAnnealer:
    """
    Sigmoid annealing for KL weight.
    Starts near 0 and gradually increases to max_weight.
    """
    def __init__(self, start_weight=0.0, max_weight=1.0,
                 center_step=500, steps_lo_to_hi=200):
        self.start_weight = start_weight
        self.max_weight = max_weight
        self.center_step = center_step
        self.steps_lo_to_hi = steps_lo_to_hi
    
    def get_weight(self, step):
        x = (step - self.center_step) / max(self.steps_lo_to_hi, 1)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return self.start_weight + (self.max_weight - self.start_weight) * sigmoid


# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_one_epoch(model, dataloader, optimizer, device, kl_weight=1.0,
                    coeff_reg_weight=0.01, clip_grad=5.0):
    """Train for one epoch."""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch (6 items from collate_fn)
        (history_batch, future_batch, neighbor_histories_batch,
         goals_batch, expanded_goals_batch, all_futures_batch) = batch
        
        # Move to device
        history_batch = history_batch.to(device)
        future_batch = future_batch.to(device)
        goals_batch = goals_batch.to(device)
        expanded_goals_batch = expanded_goals_batch.to(device)
        all_futures_batch = all_futures_batch.to(device)
        
        # history_batch shape: [B, A, ent, H, 2]
        # future_batch shape: [B, A, T, 2] (ego only)
        # goals_batch shape: [B, A, 2]
        # all_futures_batch: [B, A, ent, T, 2]
        
        optimizer.zero_grad()
        
        try:
            total_loss, loss_dict = model.train_loss(
                history_neighbors=history_batch,
                goal=goals_batch,
                future=future_batch,
                all_futures=all_futures_batch,
                kl_weight=kl_weight,
                coeff_reg_weight=coeff_reg_weight,
            )
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n[WARN] NaN/Inf loss at batch {batch_idx}, skipping...")
                continue
            
            total_loss.backward()
            
            # Gradient clipping
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            metrics.update(loss_dict)
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'nll': f"{loss_dict['nll_adjusted']:.4f}",
                'kl': f"{loss_dict['kl_loss']:.4f}",
            })
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[WARN] OOM at batch {batch_idx}, skipping...")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise
    
    return metrics.summary()


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================
@torch.no_grad()
def validate(model, dataloader, device, kl_weight=1.0):
    """Validate on held-out data."""
    model.eval()
    metrics = MetricsTracker()
    
    ade_list, fde_list = [], []
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        (history_batch, future_batch, neighbor_histories_batch,
         goals_batch, expanded_goals_batch, all_futures_batch) = batch
        
        history_batch = history_batch.to(device)
        future_batch = future_batch.to(device)
        goals_batch = goals_batch.to(device)
        expanded_goals_batch = expanded_goals_batch.to(device)
        all_futures_batch = all_futures_batch.to(device)
        
        try:
            total_loss, loss_dict = model.train_loss(
                history_neighbors=history_batch,
                goal=goals_batch,
                future=future_batch,
                all_futures=all_futures_batch,
                kl_weight=kl_weight,
            )
            
            metrics.update(loss_dict)
            
            # Compute ADE/FDE
            adjusted_preds, decoded_preds, _, _, _ = model.forward(
                history_neighbors=history_batch,
                goal=goals_batch,
                future=future_batch,
                mode=ModeKeys.EVAL,
            )
            
            ego_pred = adjusted_preds[:, :, 0, :, :]  # [B, A, T, D]
            ade = compute_ade(ego_pred, future_batch)
            fde = compute_fde(ego_pred, future_batch)
            ade_list.append(ade)
            fde_list.append(fde)
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            raise
    
    summary = metrics.summary()
    summary['ade'] = np.mean(ade_list) if ade_list else float('inf')
    summary['fde'] = np.mean(fde_list) if fde_list else float('inf')
    return summary


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    # =========================================================================
    # HYPERPARAMETERS
    # =========================================================================
    config = {
        # Data
        'data_file': 'data/combined_annotations.csv',
        'history_len': 8,
        'prediction_len': 12,
        'max_neighbors': 12,
        'batch_size': 4,
        'val_split': 0.2,
        
        # Encoder (Trajectron++ style)
        'state_dim': 2,
        'enc_rnn_dim_history': 32,
        'enc_rnn_dim_edge': 32,
        'enc_rnn_dim_future': 32,
        'edge_influence_combine': 'attention',
        
        # CVAE Latent
        'N': 1,
        'K': 25,
        'p_z_x_MLP_dims': 32,
        'q_z_xy_MLP_dims': 32,
        
        # Decoder (CfC)
        'dec_rnn_dim': 128,
        'cfc_mode': 'default',
        'cfc_backbone_units': 128,
        'cfc_backbone_layers': 1,
        'cfc_backbone_activation': 'lecun_tanh',
        'cfc_backbone_dropout': 0.0,
        'mixed_memory': True,
        
        # Kinematic
        'use_angular_velocity': True,
        'target_avg_speed': 4.087,
        'speed_tolerance': 0.15,
        'dt': 0.1,
        
        # PFM
        'num_agents': 1000,
        'pfm_k_init': 1.0,
        'pfm_repulsion_radius': 0.5,
        
        # Training
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'grad_clip': 5.0,
        'kl_weight_max': 1.0,
        'kl_anneal_center': 500,
        'kl_anneal_steps': 200,
        'coeff_reg_weight': 0.01,
        'early_stopping_patience': 15,
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,
        'checkpoint_dir': 'checkpoints',
        'model_name': 'trajectron_ltc_model',
    }
    
    print("=" * 80)
    print("  TRAJECTRON++ ENCODER — LTC/CfC DECODER — PFM HYBRID MODEL")
    print("  Training Script")
    print("=" * 80)
    print(f"\n  Device: {config['device']}")
    print(f"  CfC Mode: {config['cfc_mode']}")
    print(f"  Mixed Memory: {config['mixed_memory']}")
    print(f"  Angular Velocity: {config['use_angular_velocity']}")
    print(f"  Latent: N={config['N']}, K={config['K']} (z_dim={config['N']*config['K']})")
    print(f"  Edge Influence: {config['edge_influence_combine']}")
    print()
    
    device = torch.device(config['device'])
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("[DATA] Loading dataset...")
    data_path = os.path.join(_BASE_DIR, config['data_file'])
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        print("[INFO] Trying alternative paths...")
        alt_paths = [
            os.path.join(_BASE_DIR, 'data', 'crowds_zara02_test_cleaned.txt'),
            os.path.join(_BASE_DIR, 'data', 'crowds_zara02_test.txt'),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                data_path = alt
                print(f"[INFO] Using: {data_path}")
                break
        else:
            print("[ERROR] No data file found. Please check data/ directory.")
            return
    
    dataset = PFM_TrajectoryDataset_neighbours(
        file_path=data_path,
        history_len=config['history_len'],
        prediction_len=config['prediction_len'],
        max_neighbors=config['max_neighbors'],
    )
    
    print(f"[DATA] Dataset size: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("[ERROR] Empty dataset. Exiting.")
        return
    
    # Train/val split
    val_size = max(1, int(len(dataset) * config['val_split']))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"[DATA] Train: {train_size}, Val: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=(config['device'] == 'cuda'),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
    )
    
    # =========================================================================
    # MODEL CREATION
    # =========================================================================
    print("\n[MODEL] Creating TrajectronLTC model...")
    
    model = TrajectronLTC(
        state_dim=config['state_dim'],
        pred_state_dim=config['state_dim'],
        enc_rnn_dim_history=config['enc_rnn_dim_history'],
        enc_rnn_dim_edge=config['enc_rnn_dim_edge'],
        enc_rnn_dim_future=config['enc_rnn_dim_future'],
        edge_influence_combine=config['edge_influence_combine'],
        dec_rnn_dim=config['dec_rnn_dim'],
        N=config['N'],
        K=config['K'],
        prediction_horizon=config['prediction_len'],
        cfc_mode=config['cfc_mode'],
        cfc_backbone_units=config['cfc_backbone_units'],
        cfc_backbone_layers=config['cfc_backbone_layers'],
        cfc_backbone_activation=config['cfc_backbone_activation'],
        cfc_backbone_dropout=config['cfc_backbone_dropout'],
        mixed_memory=config['mixed_memory'],
        use_angular_velocity=config['use_angular_velocity'],
        target_avg_speed=config['target_avg_speed'],
        speed_tolerance=config['speed_tolerance'],
        dt=config['dt'],
        num_agents=config['num_agents'],
        pfm_k_init=config['pfm_k_init'],
        pfm_repulsion_radius=config['pfm_repulsion_radius'],
        p_z_x_MLP_dims=config['p_z_x_MLP_dims'],
        q_z_xy_MLP_dims=config['q_z_xy_MLP_dims'],
    ).to(device)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")
    
    # Component breakdown
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder_cell.parameters())
    pfm_params = sum(p.numel() for p in model.pfm.parameters())
    print(f"[MODEL] Encoder params: {enc_params:,}")
    print(f"[MODEL] Decoder cell params: {dec_params:,}")
    print(f"[MODEL] PFM params: {pfm_params:,}")
    
    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    kl_annealer = KLAnnealer(
        start_weight=0.0,
        max_weight=config['kl_weight_max'],
        center_step=config['kl_anneal_center'],
        steps_lo_to_hi=config['kl_anneal_steps'],
    )
    
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        verbose=True,
    )
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "=" * 80)
    print("  STARTING TRAINING")
    print("=" * 80)
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Save config
    config_path = os.path.join(
        config['checkpoint_dir'],
        f"{config['model_name']}_config.json"
    )
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[CONFIG] Saved to {config_path}")
    
    best_val_loss = float('inf')
    global_step = 0
    history_log = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Compute KL weight with annealing
        kl_weight = kl_annealer.get_weight(global_step)
        
        print(f"\n{'─' * 70}")
        print(f"  Epoch {epoch}/{config['num_epochs']}  |  "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}  |  "
              f"KL weight: {kl_weight:.4f}")
        print(f"{'─' * 70}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            kl_weight=kl_weight,
            coeff_reg_weight=config['coeff_reg_weight'],
            clip_grad=config['grad_clip'],
        )
        global_step += len(train_loader)
        
        # Validate
        val_metrics = validate(model, val_loader, device, kl_weight=kl_weight)
        
        # Scheduler step
        scheduler.step(val_metrics['total_loss'])
        
        # Print metrics
        epoch_time = time.time() - epoch_start
        print(f"\n  Train Loss: {train_metrics['total_loss']:.4f}  |  "
              f"NLL(adj): {train_metrics['nll_adjusted']:.4f}  |  "
              f"KL: {train_metrics['kl_loss']:.4f}")
        print(f"  Val   Loss: {val_metrics['total_loss']:.4f}  |  "
              f"NLL(adj): {val_metrics['nll_adjusted']:.4f}  |  "
              f"ADE: {val_metrics.get('ade', 0):.4f}  |  "
              f"FDE: {val_metrics.get('fde', 0):.4f}")
        print(f"  Coefficients — Mean: {val_metrics['coeff_mean']:.4f}  |  "
              f"Var: {val_metrics['coeff_var']:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Log
        history_log.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'kl_weight': kl_weight,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        })
        
        # Save checkpoint
        val_loss = val_metrics['total_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(
                config['checkpoint_dir'],
                f"{config['model_name']}_best.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ade': val_metrics.get('ade', None),
                'val_fde': val_metrics.get('fde', None),
                'config': config,
            }, ckpt_path)
            print(f"  ★ Best model saved to {ckpt_path}")
        
        # Periodic checkpoint
        if epoch % 5 == 0:
            ckpt_path = os.path.join(
                config['checkpoint_dir'],
                f"{config['model_name']}_epoch{epoch}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\n  *** EARLY STOPPING TRIGGERED ***")
            break
    
    # =========================================================================
    # SAVE FINAL RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)
    
    # Save training history
    history_path = os.path.join(
        config['checkpoint_dir'],
        f"{config['model_name']}_history.json"
    )
    with open(history_path, 'w') as f:
        json.dump(history_log, f, indent=2)
    print(f"[HISTORY] Saved to {history_path}")
    
    # Final model
    final_path = os.path.join(
        config['checkpoint_dir'],
        f"{config['model_name']}_final.pth"
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_epoch': epoch,
        'best_val_loss': best_val_loss,
    }, final_path)
    print(f"[FINAL] Model saved to {final_path}")
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Done!")


if __name__ == '__main__':
    main()
