import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from models.pfm_model_neighbours import PFMOnlyModel
from utils.collate_mta_pfm_neighbours import collate_fn
from torch.utils.data import DataLoader, random_split



import gc
import torch
import torch.nn as nn

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import gc
import os

# Import your dataset, model, and collate function
# from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
# from models.pfm_model import PFMOnlyModel
# from utils.collate_mta_pfm_neighbours import collate_fn

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def try_free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train_pfm_model(
    data_path,
    model_save_path,
    model_class,
    dataset_class,
    collate_fn,
    batch_size=8,
    epochs=3,
    learning_rate=1e-3,
    weight_decay=0.0,
    patience=7,
    device=None
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[TRAIN] Using device: {device}")

    # Load dataset
    dataset = dataset_class(data_path)
    print(f"[TRAIN] Loaded dataset with {len(dataset)} samples")

    # Split train/val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"[TRAIN] Split → Train: {train_size}, Val: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b,
            history_len=dataset.history_len,
            prediction_len=dataset.prediction_len,
            max_neighbors=dataset.max_neighbors
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b,
            history_len=dataset.history_len,
            prediction_len=dataset.prediction_len,
            max_neighbors=dataset.max_neighbors
        )
    )

    # Initialize model
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    # Monitor initial coefficients
    init_coeffs = model.pfm.coeff_embedding.weight.data.clone()
    print("\n[TRAIN] Initial force coefficients sample:")
    print(init_coeffs[:3])

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\n=== [EPOCH {epoch + 1}/{epochs}] ===")
        model.train()
        epoch_loss = 0.0

        for batch_idx, (history_batch, future_batch, neighbor_batch, goals_batch, expanded_goals_batch, all_futures_batch) in enumerate(train_loader):
            # Move to device
            history_batch = history_batch.to(device)
            future_batch = future_batch.to(device)
            expanded_goals_batch = expanded_goals_batch.to(device)

            # Extract ego future (first entity in all_futures)
            # Future shape is [B, A, T, 2], but we need ego only

            optimizer.zero_grad()

            # Forward pass: model expects [B, A, N_entities, H, 2] and [B, A, N_entities, 2]
            pred, coeff_mean, coeff_var = model(history_batch, expanded_goals_batch)

            # pred shape: [B, A, 12, 2]
            # future_batch shape: [B, A, 12, 2] (ego only)
            loss = criterion(pred, future_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Logging
            if batch_idx % 10 == 0:
                grads = model.pfm.coeff_embedding.weight.grad
                grad_norm = grads.norm().item() if grads is not None else 0.0
                print(f"[E{epoch+1} B{batch_idx}/{len(train_loader)}] "
                      f"Loss={loss.item():.4f}, GradNorm={grad_norm:.6f}, "
                      f"CoeffMean={coeff_mean.item():.4f}")
                print(f"  Coeff sample: {model.pfm.coeff_embedding.weight.data[:3]}")

            # Memory cleanup
            if batch_idx % 20 == 0:
                try_free_cuda()

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        print(f"[EPOCH {epoch + 1}] Train Loss: {avg_train_loss:.6f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vhist, vfut, vneigh, vgoals, vexp_goals, vall_fut in val_loader:
                vhist = vhist.to(device)
                vfut = vfut.to(device)
                vexp_goals = vexp_goals.to(device)

                vpred, _, _ = model(vhist, vexp_goals)
                val_loss += criterion(vpred, vfut).item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        print(f"[VAL] Validation Loss: {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict()}, model_save_path)
            print(f"✅ Saved new best model → Val Loss: {avg_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"⚠️ EarlyStopping: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⏹ Early stopping triggered.")
            break

        try_free_cuda()

    print(f"\n🏁 Training Complete. Best model saved to: {model_save_path}")
    print(f"Final force coefficients sample:")
    print(model.pfm.coeff_embedding.weight.data[:3])

    return train_losses, val_losses


if __name__ == "__main__":
    train_pfm_model(
        data_path="data\combined_annotations.csv",
        model_save_path="checkpoints/pfm_learnable_neighbours_model.pth",
        model_class=PFMOnlyModel,
        dataset_class=PFM_TrajectoryDataset_neighbours,
        collate_fn=collate_fn,
        batch_size=1,  # Start with 1 for large scenes
        epochs=1,
        learning_rate=1e-3,
        patience=7
    )
