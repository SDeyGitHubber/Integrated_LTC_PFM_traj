import torch
import gc
from torch import nn
from torch.utils.data import DataLoader, random_split
from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
from utils.collate_mta_pfm_neighbours import collate_fn
from utils.speed_utils import calculate_speed, check_speed_violations
from models.mta_pfm_model_neighbours import CheckpointedIntegratedMTAPFM_neighbours

import os
import math
import gc
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt

# from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
# from utils.collate_mta_pfm_neighbours import collate_fn
# from models.mta_pfm_model_neighbours import CheckpointedIntegratedMTAPFM_neighbours

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def ensure_batch_dim(tensor):
    if tensor is None:
        return None
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def try_free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def clean_tensor(tensor):
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("mta_pfm_loss_plot.png")
    plt.show()

def train_mta_pfm_model(
    data_path,
    model_save_path,
    model_class=CheckpointedIntegratedMTAPFM_neighbours,
    dataset_class=PFM_TrajectoryDataset_neighbours,
    collate_fn=collate_fn,
    batch_size=1,
    epochs=1,
    learning_rate=0.0001,
    weight_decay=0.0,
    patience=7,
    accumulation_steps=4,
    device=None,
    max_agents_per_forward=32,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TRAIN] Using device: {device}")

    dataset = dataset_class(data_path)
    full_len = len(dataset)
    reduced_len = int(full_len * 0.2)
    dataset = Subset(dataset, list(range(reduced_len)))
    print(f"[TRAIN] Using 20% subset of dataset: {reduced_len}/{full_len} samples")

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"[TRAIN] Split → Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b,
            history_len=dataset.dataset.history_len,
            prediction_len=dataset.dataset.prediction_len,
            max_neighbors=dataset.dataset.max_neighbors,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b,
            history_len=dataset.dataset.history_len,
            prediction_len=dataset.dataset.prediction_len,
            max_neighbors=dataset.dataset.max_neighbors,
        ),
    )

    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    print("\n=== [TRAIN] Speed Constraints ===")
    print(f"    Target Avg Speed: {model.target_avg_speed:.4f}")
    print(f"    Allowed Range: [{model.min_speed:.4f}, {model.max_speed:.4f}]")
    print(f"    Tolerance: ±{model.speed_tolerance * 100:.1f}%")

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\n=== [EPOCH {epoch + 1}/{epochs}] ===")
        model.train()
        epoch_loss = 0.0

        total_batches = len(train_loader)
        # FIX: Unpack 6 values instead of 5
        for batch_idx, (history_neighbors, future, neighbor_histories, goals, expanded_goals, all_futures) in enumerate(train_loader):
            history_neighbors = ensure_batch_dim(history_neighbors).to(device)
            future = ensure_batch_dim(future).to(device)
            neighbor_histories = ensure_batch_dim(neighbor_histories).to(device)
            expanded_goals = ensure_batch_dim(expanded_goals).to(device)

            history_neighbors = clean_tensor(history_neighbors)
            future = clean_tensor(future)
            neighbor_histories = clean_tensor(neighbor_histories)
            expanded_goals = clean_tensor(expanded_goals)

            A = history_neighbors.shape[1]
            if A == 0:
                continue

            chunk_size = min(max_agents_per_forward, A)
            processed_successfully = False

            while chunk_size >= 1 and not processed_successfully:
                try:
                    num_chunks = math.ceil(A / chunk_size)
                    optimizer.zero_grad()
                    batch_epoch_loss = 0.0

                    for chunk_idx in range(num_chunks):
                        s, e = chunk_idx * chunk_size, min((chunk_idx + 1) * chunk_size, A)
                        hist_chunk = history_neighbors[:, s:e, :, :, :]
                        fut_chunk = future[:, s:e, :, :]
                        exp_goal_chunk = expanded_goals[:, s:e, :, :]

                        hist_chunk = clean_tensor(hist_chunk)
                        fut_chunk = clean_tensor(fut_chunk)
                        exp_goal_chunk = clean_tensor(exp_goal_chunk)

                        adjusted_preds_chunk, decoded_preds_chunk, coeff_mean, coeff_var = model(hist_chunk, exp_goal_chunk)
                        ego_pred_chunk = adjusted_preds_chunk[:, :, 0, :, :]

                        loss_chunk_raw = criterion(ego_pred_chunk, fut_chunk)
                        loss_to_backward = loss_chunk_raw / float(num_chunks)
                        loss_to_backward.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        batch_epoch_loss += loss_chunk_raw.item()
                        optimizer.step()

                    epoch_loss += batch_epoch_loss
                    processed_successfully = True
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[OOM] Batch {batch_idx}: reducing chunk size {chunk_size} → {max(1, chunk_size // 2)}")
                        try_free_cuda()
                        chunk_size = max(1, chunk_size // 2)
                        continue
                    else:
                        raise e

            percent = 100 * (batch_idx + 1) / total_batches
            print(f">>> {percent:.2f}% of training for epoch {epoch + 1} completed (batch {batch_idx + 1}/{total_batches}) <<<")

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        print(f"[EPOCH {epoch + 1}] Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # FIX: Unpack 6 values instead of 5
            for vhist_neighbors, vfuture, vneigh_histories, vgoals, vexp_goals, vall_futures in val_loader:
                preds, _, _, _ = model(vhist_neighbors.to(device), vexp_goals.to(device))
                val_loss += criterion(preds[:, :, 0, :, :], vfuture.to(device)).item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        print(f"[VAL] Validation Loss: {avg_val_loss:.6f}")

        epoch_weight_path = f"{model_save_path}_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), epoch_weight_path)
        print(f"💾 Saved model weights: {epoch_weight_path}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"✅ New best model: Epoch {epoch + 1}, Val Loss {avg_val_loss:.6f}")

        try_free_cuda()

    print(f"\n🏁 Training Complete. Final weights saved to: {model_save_path}")
    plot_loss(train_losses, val_losses)
    # main.py

import os
import torch
# Assuming train.py is directly importable, or its path is adjusted

# --- Configuration ---
# You need to adjust these paths and values based on your setup.
DATA_FILE_PATH = "C:\ShouvikDey\iDataHub_Intern\iDataHubIntern_running\data\combined_annotations.csv"  # <--- REPLACE with your actual data file path
MODEL_SAVE_DIR = "checkpoints"
MODEL_PREFIX = "mta_pfm_model_v1"

# --- Hyperparameters ---
CONFIG = {
    "data_path": DATA_FILE_PATH,
    "model_save_path": os.path.join(MODEL_SAVE_DIR, MODEL_PREFIX),
    "batch_size": 1,
    "epochs": 5,  # Set a reasonable number of epochs
    "learning_rate": 0.0001,
    "max_agents_per_forward": 32, # Crucial for managing GPU memory (OOM error handling is present)
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

def main():
    """
    Sets up the environment and calls the main training function.
    """
    # Create the directory for saving model checkpoints if it doesn't exist
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created model save directory: {MODEL_SAVE_DIR}")

    print("--- Starting MTA-PFM Model Training ---")
    print(f"Configuration: {CONFIG}")

    # Check if the data file exists
    if not os.path.exists(CONFIG["data_path"]):
        print(f"\n[ERROR] Data file not found at: {CONFIG['data_path']}")
        print("Please check the DATA_FILE_PATH variable and ensure your trajectory data file exists.")
        return

    # Call the main training loop
    try:
        train_mta_pfm_model(**CONFIG)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An error occurred during training: {e}")
        # Optionally, you can add more specific error handling here

    print("\n--- Training Process Concluded ---")

if __name__ == "__main__":
    main()