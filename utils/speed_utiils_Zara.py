import torch
import torch.nn as nn


def calculate_speed(trajectories):
    """
    Calculate average speed from trajectory data
    Args:
        trajectories: [B, A, T, 2] tensor where T is time steps
    Returns:
        average_speed: scalar tensor
    """
    B, A, T, D = trajectories.shape

    # Filter out padding (0,0) points
    non_zero_mask = ~((trajectories[:, :, :, 0] == 0) & (trajectories[:, :, :, 1] == 0))

    # Calculate displacement between consecutive time steps
    # trajectories[:, :, 1:] - trajectories[:, :, :-1] gives [B, A, T-1, 2]
    displacements = trajectories[:, :, 1:] - trajectories[:, :, :-1]

    # Calculate distances (speeds) for each time step
    distances = torch.norm(displacements, dim=-1)  # [B, A, T-1]

    # Apply mask to ignore padded transitions (both current and next point should be non-zero)
    valid_transitions = non_zero_mask[:, :, :-1] & non_zero_mask[:, :, 1:]

    # Only consider valid (non-padded) transitions
    valid_distances = distances * valid_transitions.float()

    # Calculate average speed (sum of valid distances / number of valid transitions)
    total_distance = valid_distances.sum()
    total_transitions = valid_transitions.sum()

    if total_transitions > 0:
        avg_speed = total_distance / total_transitions
    else:
        avg_speed = torch.tensor(0.0, device=trajectories.device)

    return avg_speed

# === HELPER FUNCTION TO CHECK VIOLATIONS ===
def check_speed_violations(predictions, history, min_speed, max_speed):
    """
    Count the number of speed constraint violations in predictions
    """
    B, A, T, D = predictions.shape
    last_pos = history[:, :, -1, :]  # [B, A, 2]
    current_pos = last_pos.clone()

    violations = 0

    for t in range(T):
        displacement = predictions[:, :, t] - current_pos
        speeds = torch.norm(displacement, dim=-1)

        # Count violations (excluding zero speeds from padding)
        non_zero_mask = speeds > 0
        too_fast = (speeds > max_speed) & non_zero_mask
        too_slow = (speeds < min_speed) & non_zero_mask

        violations += (too_fast | too_slow).sum().item()
        current_pos = predictions[:, :, t].clone()

    return violations