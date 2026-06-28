# Speed.py

import torch
import torch.nn as nn

def calculate_speed(trajectory):
    """Calculate average speed from trajectory tensor [B, A, T, 2]"""
    diffs = trajectory[:, :, 1:, :] - trajectory[:, :, :-1, :]  # Frame-to-frame differences
    speeds = torch.norm(diffs, dim=-1)  # [B, A, T-1]
    valid_mask = speeds > 0
    if valid_mask.sum() > 0:
        avg_speed = speeds[valid_mask].mean()
    else:
        avg_speed = torch.tensor(0.0)
    return avg_speed

def check_speed_violations(predictions, history, min_speed, max_speed):
    """Count speed violations in predicted trajectories"""
    # Get last history position
    last_pos = history[:, :, -1:, :]  # [B, A, 1, 2]

    # Combine last history with predictions for speed calculation
    full_traj = torch.cat([last_pos, predictions], dim=2)  # [B, A, T+1, 2]

    # Calculate speeds
    diffs = full_traj[:, :, 1:, :] - full_traj[:, :, :-1, :]
    speeds = torch.norm(diffs, dim=-1)  # [B, A, T]

    # Count violations
    violations = ((speeds < min_speed) | (speeds > max_speed)).sum().item()
    return violations