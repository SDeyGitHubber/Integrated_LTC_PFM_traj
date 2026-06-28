import torch
import gc
from torch import nn
from torch.utils.data import DataLoader, random_split
from utils.collate import collate_fn
from datasets.pfm_trajectory_dataset import PFM_TrajectoryDataset
from utils.speed_utils import calculate_speed, check_speed_violations
from models.mta_pfm_model import CheckpointedIntegratedMTAPFMModel


def train_mta_pfm_model(
    data_path,
    model_save_path,
    model_class,
    dataset_class,
    collate_fn,
    batch_size=32,
    epochs=50,
    learning_rate=0.001,
    weight_decay=0.0,
    patience=7,
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # === DATA LOADING WITH VALIDATION SPLIT ===
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

    # === MODEL, OPTIMIZER, SCHEDULER, LOSS ===
    model = model_class().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    print(f"\n🎯 Speed Constraints Enabled:")
    print(f"   Target Avg Speed: {model.target_avg_speed:.4f}")
    print(f"   Allowed range: [{model.min_speed:.4f}, {model.max_speed:.4f}]")
    print(f"   Tolerance: ±{model.speed_tolerance * 100:.1f}%\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # --------- TRAINING ---------
        model.train()
        epoch_loss = 0.0
        epoch_gt_speeds, epoch_pred_speeds, epoch_hist_speeds, epoch_violations = [], [], [], []

        for batch_idx, (history, future, neighbors, goal) in enumerate(train_loader):
            history, future, neighbors = (
                history.to(device),
                future.to(device),
                neighbors.to(device),
            )
            goal = future[:, :, -1, :].clone()

            optimizer.zero_grad()
            pred, coeff_mean, coeff_var = model(history, neighbors, goal)
            loss = criterion(pred, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                gt_speed = calculate_speed(future)
                pred_speed = calculate_speed(pred)
                hist_speed = calculate_speed(history)

                epoch_gt_speeds.append(gt_speed.item())
                epoch_pred_speeds.append(pred_speed.item())
                epoch_hist_speeds.append(hist_speed.item())
                violation_count = check_speed_violations(
                    pred, history, model.min_speed, model.max_speed
                )
                epoch_violations.append(violation_count)

            if batch_idx % 50 == 0:
                print(
                    f"[{batch_idx}] Loss: {loss:.4f} | k_att1 μ={coeff_mean:.2f}, σ²={coeff_var:.2f}"
                )
                print(
                    f"    Speeds - GT: {gt_speed:.3f}, Pred: {pred_speed:.3f}, Hist: {hist_speed:.3f}"
                )
                print(f"    Speed Violations: {violation_count}")

        avg_train_loss = epoch_loss / len(train_loader)

        # --------- VALIDATION ---------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for history, future, neighbors, goal in val_loader:
                history, future, neighbors = (
                    history.to(device),
                    future.to(device),
                    neighbors.to(device),
                )
                goal = future[:, :, -1, :].clone()

                pred, _, _ = model(history, neighbors, goal)
                loss = criterion(pred, future)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"\n=== Epoch {epoch+1}/{epochs} SUMMARY ===\n"
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
        )
        avg_gt_speed = sum(epoch_gt_speeds) / len(epoch_gt_speeds)
        avg_pred_speed = sum(epoch_pred_speeds) / len(epoch_pred_speeds)
        avg_hist_speed = sum(epoch_hist_speeds) / len(epoch_hist_speeds)
        total_violations = sum(epoch_violations)
        samples = len(train_loader) * batch_size * 5 * 12
        constraint_compliance = (1 - total_violations / samples) * 100 if samples > 0 else 0

        print(f"Avg Speeds: Hist={avg_hist_speed:.4f}, GT={avg_gt_speed:.4f}, Pred={avg_pred_speed:.4f}")
        print(f"Speed Error: {abs(avg_gt_speed - avg_pred_speed):.4f}")
        print(f"Violations: {total_violations} | Constraint Compliance: {constraint_compliance:.2f}%")
        print("=" * 40)

        # Step LR scheduler
        scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict()}, model_save_path)
            print(f"✅ New best model saved at epoch {epoch+1} with Val Loss {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️ EarlyStopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⏹ Early stopping triggered.")
            break

        # Clear cache and garbage collect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    torch.autograd.set_detect_anomaly(False)
    print(f"Training completed. Best model saved to {model_save_path}")


if __name__ == "__main__":
    train_mta_pfm_model(
        data_path="data/combined_annotations.csv",
        model_save_path="checkpoint/mta_pfm_trajectory_model_own.pth",
        model_class=CheckpointedIntegratedMTAPFMModel,
        dataset_class=PFM_TrajectoryDataset,
        collate_fn=collate_fn,
        batch_size=8,
        epochs=80,
        patience=7,
    )
