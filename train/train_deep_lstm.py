import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from utils.collate import collate_fn
from datasets.pfm_trajectory_dataset import PFM_TrajectoryDataset
from models.deep_lstm_model import DeepLSTMModel
from utils.speed_utils import calculate_speed, check_speed_violations


def train_mta_model(
    data_path,
    model_save_path,
    model_class=DeepLSTMModel,
    dataset_class=PFM_TrajectoryDataset,
    collate_fn=collate_fn,
    batch_size=32,
    epochs=80,
    learning_rate=0.001,
    weight_decay=0.0,
    patience=7,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and split into train/val
    dataset = dataset_class(data_path)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Instantiate model, optimizer, criterion, scheduler
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    print(f"\n🎯 Speed Constraints Enabled:")
    print(f"   Target Avg Speed: {model.target_avg_speed:.4f}")
    print(f"   Allowed range: [{model.min_speed:.4f}, {model.max_speed:.4f}]")
    print(f"   Tolerance: ±{model.speed_tolerance*100:.1f}%\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (history, future, neighbors, goal) in enumerate(train_loader):
            history, future = history.to(device), future.to(device)

            optimizer.zero_grad()
            preds, _, _ = model(history)
            loss = criterion(preds, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 50 == 0:
                gt_spd = calculate_speed(future).item()
                pred_spd = calculate_speed(preds).item()
                hist_spd = calculate_speed(history).item()
                violations = check_speed_violations(preds, history, model.min_speed, model.max_speed)

                print(
                    f"[E{epoch+1} B{batch_idx}] TrainLoss={loss:.4f} | "
                    f"Speeds - GT: {gt_spd:.3f}, Pred: {pred_spd:.3f}, Hist: {hist_spd:.3f} | "
                    f"Violations: {violations}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for history, future, neighbors, goal in val_loader:
                history, future = history.to(device), future.to(device)
                preds, _, _ = model(history)
                loss = criterion(preds, future)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

        # Step LR scheduler
        scheduler.step(avg_val_loss)

        # Early stopping and checkpoint saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict()}, model_save_path)
            print(f"✅ New best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"⚠️ EarlyStopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⏹ Early stopping triggered.")
            break

    print(f"Training completed. Best model saved to {model_save_path}")


if __name__ == "__main__":
    train_mta_model(
    data_path ="data/combined_annotations.csv",
    model_save_path="checkpoints/mta_trajectory_model_own.pth",
    model_class=DeepLSTMModel,
    dataset_class=PFM_TrajectoryDataset,
    collate_fn=collate_fn,
    batch_size=32,
    epochs=80,
    learning_rate=0.001,
    weight_decay=0.0,
    patience=7,
    device=None,
    )