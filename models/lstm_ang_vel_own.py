import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class VelocityBasedLSTM(nn.Module):
    """
    Standard residual-velocity LSTM model, with omega_avg hard-coded to zero.
    The predicted output per step is a residual correction to speed and angular velocity.
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dt=0.25,
                 target_avg_speed=5.15, speed_tolerance=0.15):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dt = dt
        self.target_avg_speed = target_avg_speed
        self.speed_tolerance = speed_tolerance
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)

        self.input_embed = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 2)  # Output: (delta_v, delta_omega)
        self.postprocess = nn.Linear(2, 2)

    def compute_velocity_features(self, history):
        """
        Compute v_avg, omega_avg (forced to zero), and last heading from input history.
        """
        B, A, H, _ = history.shape
        device = history.device

        displacements = history[:, :, 1:, :] - history[:, :, :-1, :]
        distances = torch.norm(displacements, dim=-1)
        velocities = distances / self.dt
        dx = displacements[:, :, :, 0]
        dy = displacements[:, :, :, 1]
        theta = torch.atan2(dy, dx)

        # HARD-CODED: Force omega_avg to zero
        omega_avg = torch.zeros(B, A, device=device)

        valid_mask_v = velocities > 0.1
        v_avg = torch.where(
            valid_mask_v.any(dim=2),
            velocities.sum(dim=2) / (valid_mask_v.sum(dim=2).float() + 1e-6),
            torch.full((B, A), self.target_avg_speed, device=device)
        )
        theta_last = theta[:, :, -1] if H > 1 else torch.zeros(B, A, device=device)
        return v_avg, omega_avg, theta_last

    def integrate_velocities_to_positions(self, velocity_residuals, v_avg, omega_avg,
                                          theta_last, last_pos):
        """
        Integrate delta_v, delta_omega residuals, with omega_avg=0, to prediction points.
        """
        B, A, T, _ = velocity_residuals.shape
        positions = torch.zeros_like(velocity_residuals)
        current_pos = last_pos.clone()
        current_theta = theta_last.clone()
        for t in range(T):
            delta_v = velocity_residuals[:, :, t, 0]   # Residual for speed
            delta_omega = velocity_residuals[:, :, t, 1] # Residual for angular velocity

            v_pred = v_avg + delta_v
            omega_pred = omega_avg + delta_omega # omega_avg is always zero here

            v_pred = torch.clamp(v_pred, self.min_speed, self.max_speed)
            current_theta = current_theta + omega_pred * self.dt
            current_theta = torch.atan2(torch.sin(current_theta), torch.cos(current_theta))

            dx = v_pred * torch.cos(current_theta) * self.dt
            dy = v_pred * torch.sin(current_theta) * self.dt
            current_pos = current_pos + torch.stack([dx, dy], dim=-1)
            positions[:, :, t] = current_pos
        return positions

    def forward(self, history_neighbors, expanded_goals=None, neighbors=None):
        """
        Forward: output delta_v, delta_omega, and integrate using v_avg, omega_avg=0.
        """
        # Accept either 5D input (B, A, N, H, D) or 4D input (A, N, H, D) / (B, A, H, D)
        if history_neighbors.dim() == 5:
            # Expected: (B, A, N, H, D) - extract ego (neighbor index 0)
            history = history_neighbors[:, :, 0, :, :]
        elif history_neighbors.dim() == 4:
            # Distinguish between (B, A, H, D) and (A, N, H, D)
            # If shape[2] is small (< 20), it's likely N (neighbors), so this is (A, N, H, D)
            # If shape[2] is large (>= 20) OR shape[0] == 1, it's likely H (history) or batch, so (B, A, H, D)
            if history_neighbors.shape[2] < 20 and history_neighbors.shape[0] > 1:
                # Likely (A, N, H, D) - add batch dimension then extract ego
                history_neighbors = history_neighbors.unsqueeze(0)
                history = history_neighbors[:, :, 0, :, :]
            else:
                # Likely (B, A, H, D) - already has batch, no neighbor dimension
                history = history_neighbors
        else:
            raise ValueError(f"Expected history_neighbors to be 4D or 5D, got {history_neighbors.dim()}D")
        B, A, H, D = history.shape
        device = history.device

        v_avg, omega_avg, theta_last = self.compute_velocity_features(history)

        hist_flat = history.reshape(B * A, H, D)
        hist_embedded = self.input_embed(hist_flat)
        _, (h_n, c_n) = self.encoder(hist_embedded)

        T = 12  # Number of prediction steps
        velocity_residuals = []
        input_decoder = torch.zeros(B * A, 1, self.hidden_size, device=device)
        hx = (h_n, c_n)
        for t in range(T):
            out, hx = self.decoder(input_decoder, hx)
            step_output = self.output(out.squeeze(1))
            step_output = self.postprocess(step_output)
            velocity_residuals.append(step_output.view(B * A, 1, 2))
            # Next decoder input: feedback last decoder hidden, or keep zeros for simplicity

        velocity_residuals = torch.cat(velocity_residuals, dim=1)
        velocity_residuals = velocity_residuals.view(B, A, T, 2)
        positions = self.integrate_velocities_to_positions(
            velocity_residuals, v_avg, omega_avg, theta_last, history[:, :, -1, :]
        )
        return positions, velocity_residuals, None

DeepLSTMVelocityModel = VelocityBasedLSTM