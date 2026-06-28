import torch
import torch.nn as nn
from models.mta_model_Zara import DeepLSTMModel
from datasets.pfm_trajectory_dataset_zara import PFM_TrajectoryDataset_Zara
from utils.collate import collate_fn
from utils.speed_utiils_Zara import check_speed_violations, calculate_speed


def train_mta_model(
    data_path,
    model_save_path,
    model_class,
    dataset_class,
    collate_fn,
    batch_size=32,
    epochs=3,
    learning_rate=0.001,
    weight_decay=0.0,
    device=None
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)

    # Load dataset and dataloader
    dataset = dataset_class(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model, optimizer, criterion
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    print(f"\n🎯 Speed Constraints Enabled:")
    print(f"   Target Avg Speed: {model.target_avg_speed:.4f}")
    print(f"   Allowed range: [{model.min_speed:.4f}, {model.max_speed:.4f}]")
    print(f"   Tolerance: ±{model.speed_tolerance * 100:.1f}%\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_gt_speeds, epoch_pred_speeds, epoch_hist_speeds, epoch_violations = [], [], [], []

        for batch_idx, (history, future, neighbors, goal) in enumerate(dataloader):
            history = history.to(device)
            future = future.to(device)
            neighbors = neighbors.to(device)
            goal = future[:, :, -1, :].clone().to(device)

            optimizer.zero_grad()
            pred, _, _ = model(history, neighbors, goal)
            loss = criterion(pred, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                gt_speed = calculate_speed(future)
                pred_speed = calculate_speed(pred)
                hist_speed = calculate_speed(history)
                violation_count = check_speed_violations(pred, history, model.min_speed, model.max_speed)

                epoch_gt_speeds.append(gt_speed.item())
                epoch_pred_speeds.append(pred_speed.item())
                epoch_hist_speeds.append(hist_speed.item())
                epoch_violations.append(violation_count)

            if batch_idx % 50 == 0:
                print(f"[{batch_idx}] Loss: {loss:.4f}")
                print(f"    Speeds - GT: {gt_speed:.3f}, Pred: {pred_speed:.3f}, Hist: {hist_speed:.3f}")
                print(f"    Speed Violations: {violation_count}")

        avg_gt_speed = sum(epoch_gt_speeds) / len(epoch_gt_speeds)
        avg_pred_speed = sum(epoch_pred_speeds) / len(epoch_pred_speeds)
        avg_hist_speed = sum(epoch_hist_speeds) / len(epoch_hist_speeds)
        total_violations = sum(epoch_violations)

        print(f"\n=== EPOCH {epoch+1} SUMMARY ===")
        print(f"Avg Loss: {epoch_loss/len(dataloader):.4f}")
        print(f"Avg Speeds: Hist={avg_hist_speed:.4f}, GT={avg_gt_speed:.4f}, Pred={avg_pred_speed:.4f}")
        print(f"Speed Error: {abs(avg_gt_speed - avg_pred_speed):.4f}")
        print(f"Violations: {total_violations} | Constraint Compliance: {(1 - total_violations/(len(dataloader)*batch_size*5*12))*100:.2f}%")
        print("=" * 40)

    torch.autograd.set_detect_anomaly(False)

    # Save model checkpoint
    print("\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'loss': epoch_loss / len(dataloader),
        'speed_constraints': {
            'target_avg_speed': model.target_avg_speed,
            'min_speed': model.min_speed,
            'max_speed': model.max_speed,
            'tolerance': model.speed_tolerance
        },
        'final_avg_speeds': {
            'historical': avg_hist_speed,
            'ground_truth': avg_gt_speed,
            'predicted': avg_pred_speed
        }
    }, model_save_path)
    print(f"✅ Model saved at {model_save_path} with speed constraints!")

if __name__ == "__main__":
    train_mta_model(
        data_path="data/crowds_zara02_test_cleaned.txt",
        model_save_path="/checkpoints/mta_trajectory_model_Zara.pth",
        model_class=DeepLSTMModel,
        dataset_class=PFM_TrajectoryDataset_Zara,
        collate_fn=collate_fn,
        batch_size=32,
        epochs=5
    )
