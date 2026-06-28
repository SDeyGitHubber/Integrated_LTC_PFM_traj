# lstm_velocity_model.py
"""
Velocity-based LSTM model for trajectory prediction.
Predicts velocity residuals (Δv, Δω) instead of positions.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class VelocityBasedLSTM(nn.Module):
    """
    LSTM model that predicts velocity residuals and integrates them to positions.

    Architecture:
    1. Encoder LSTM: Encodes position history
    2. Decoder LSTM: Predicts velocity residuals (Δv, Δω) at each timestep
    3. Kinematic Integration: Converts velocity residuals to positions
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dt=0.25,
                 target_avg_speed=5.15, speed_tolerance=0.15):
        """
        Args:
            input_size: Input dimension (2 for x,y positions)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dt: Time step between frames (0.25 for 4 FPS)
            target_avg_speed: Average speed from dataset calculation
            speed_tolerance: Speed constraint tolerance (±15%)
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dt = dt
        self.target_avg_speed = target_avg_speed
        self.speed_tolerance = speed_tolerance
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)

        # Network layers
        self.input_embed = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # CHANGE: Output now represents (Δv, Δω) velocity residuals
        self.output = nn.Linear(hidden_size, 2)  # 2D: [Δv, Δω]
        self.postprocess = nn.Linear(2, 2)

    def compute_velocity_features(self, history):
        """
        STEP 1: Compute velocity features from position history.

        Extracts:
        - Vi = √((xi+1 - xi)² + (yi+1 - yi)²) / dt
        - θi = arctan((yi+1 - yi) / (xi+1 - xi))
        - ωi = (θi+1 - θi) / dt

        Args:
            history: [B, A, H, 2] position history

        Returns:
            v_avg: [B, A] average linear velocity
            omega_avg: [B, A] average angular velocity
            theta_last: [B, A] last heading angle
        """
        B, A, H, _ = history.shape
        device = history.device

        # Compute consecutive displacements
        displacements = history[:, :, 1:, :] - history[:, :, :-1, :]  # [B, A, H-1, 2]

        # Linear velocities: Vi = ||displacement|| / dt       #linear velocity, hard code w avg to 0, vel residuals to 0 implies history is same
        distances = torch.norm(displacements, dim=-1)  # [B, A, H-1]
        velocities = distances / self.dt

        # Heading angles: θi = arctan2(dy, dx)
        dx = displacements[:, :, :, 0]
        dy = displacements[:, :, :, 1]
        theta = torch.atan2(dy, dx)  # [B, A, H-1]

        # Angular velocities: ωi = (θi+1 - θi) / dt
        omega_avg = torch.zeros(B, A, device=device)

        # Compute average linear velocity
        valid_mask_v = velocities > 0.1
        v_avg = torch.where(
            valid_mask_v.any(dim=2),
            velocities.sum(dim=2) / (valid_mask_v.sum(dim=2).float() + 1e-6),
            torch.full((B, A), self.target_avg_speed, device=device)
        )

        # Last heading angle
        theta_last = theta[:, :, -1] if H > 1 else torch.zeros(B, A, device=device)

        return v_avg, omega_avg, theta_last

    def integrate_velocities_to_positions(self, velocity_residuals, v_avg, omega_avg,
                                          theta_last, last_pos):
        """
        STEP 4: Integrate velocity residuals to positions using kinematic model.

        For each timestep:
        1. Vi_t = Vavg + Δvi_t
        2. ωi_t = ωavg + Δωi_t
        3. θi_t+1 = θi_t + ωi_t * dt
        4. Px_t+1 = Px_t + Vi_t * cos(θi_t) * dt
        5. Py_t+1 = Py_t + Vi_t * sin(θi_t) * dt

        Args:
            velocity_residuals: [B, A, T, 2] where [:,:,:,0]=Δv, [:,:,:,1]=Δω
            v_avg: [B, A] average velocity from history
            omega_avg: [B, A] average angular velocity from history
            theta_last: [B, A] last heading from history
            last_pos: [B, A, 2] last known position

        Returns:
            positions: [B, A, T, 2] integrated positions
        """
        B, A, T, _ = velocity_residuals.shape
        positions = torch.zeros_like(velocity_residuals)

        current_pos = last_pos.clone()  # [B, A, 2]
        current_theta = theta_last.clone()  # [B, A]

        for t in range(T):
            # Extract velocity residuals for this timestep
            delta_v = velocity_residuals[:, :, t, 0]  # [B, A]
            delta_omega = velocity_residuals[:, :, t, 1]  # [B, A]

            # STEP 3: Compute predicted velocities
            # Vi_t+1 = Vavg + Δvi_t
            v_pred = v_avg + delta_v

            # ωi_t+1 = ωavg + Δωi_t
            omega_pred = omega_avg + delta_omega

            # Apply speed constraints
            v_pred = torch.clamp(v_pred, self.min_speed, self.max_speed)

            # STEP 4: Update heading
            # θi_t+1 = θi_t + ωi_t+1 * dt
            current_theta = current_theta + omega_pred * self.dt
            # Wrap to [-π, π]
            current_theta = torch.atan2(torch.sin(current_theta), torch.cos(current_theta))

            # STEP 4: Update position using kinematic model
            # Px_t+1 = Px_t + Vi_t+1 * cos(θi_t) * dt
            # Py_t+1 = Py_t + Vi_t+1 * sin(θi_t) * dt
            dx = v_pred * torch.cos(current_theta) * self.dt
            dy = v_pred * torch.sin(current_theta) * self.dt
            current_pos = current_pos + torch.stack([dx, dy], dim=-1)

            positions[:, :, t] = current_pos

        return positions

    def forward_encoder(self, x):
        """Encoder forward pass (for gradient checkpointing)."""
        return self.encoder(x)

    def forward_decoder_step(self, input_step, hx):
        """Decoder single step (for gradient checkpointing)."""
        return self.decoder(input_step, hx)

    def forward(self, history_neighbors, expanded_goals=None, neighbors=None):
        """
        Main forward pass.

        Args:
            history_neighbors: [B, A, ent, H, 2] - from dataloader
            expanded_goals: [B, A, ent, 2] - from dataloader (optional, not used)
            neighbors: Not used (kept for API compatibility)

        Returns:
            positions: [B, A, 12, 2] - predicted positions
            None, None - placeholders for API compatibility
        """
        # Extract ego history from history_neighbors
        # history_neighbors shape: [B, A, ent, H, 2]
        # ego history is at index 0: [B, A, 0, H, 2]
        history = history_neighbors[:, :, 0, :, :]  # [B, A, H, 2]

        B, A, H, D = history.shape
        device = history.device

        # STEP 1: Compute velocity features from history
        v_avg, omega_avg, theta_last = self.compute_velocity_features(history)

        # Reshape for LSTM: [B*A, H, D]
        hist_flat = history.reshape(B * A, H, D)
        hist_embedded = self.input_embed(hist_flat)

        # Encoder with gradient checkpointing
        _, (h_n, c_n) = self.encoder(hist_embedded)

        # STEP 2: Decoder predicts velocity residuals
        velocity_residuals = []
        input_decoder = torch.zeros(B * A, 1, self.hidden_size, device=device)

        hx = (h_n, c_n)

        # Autoregressive decoding
        for t in range(12):
            # Input: previous velocity residual (or zero for t=0)
            out, hx = self.decoder(input_decoder, hx)
            step_output = self.output(out.squeeze(1))
            step_output = self.postprocess(step_output)
            velocity_residuals.append(step_output.view(B*A, 1, 2))

        # Reshape back to [B, A, T, 2]
        velocity_residuals = torch.cat(velocity_residuals, dim = 1)
        velocity_residuals = velocity_residuals.view(B,A,12,2)
        positions = self.integrate_velocities_to_positions(
            velocity_residuals, v_avg, omega_avg, theta_last, history[:, :,-1,:]
        )

        return positions, velocity_residuals, None


# Alias for backward compatibility
DeepLSTMVelocityModel = VelocityBasedLSTM
