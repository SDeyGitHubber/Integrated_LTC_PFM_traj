"""
Liquid Time-Constant Network (LTC) for Velocity-Based Trajectory Prediction

This module implements a continuous-time dynamical system for pedestrian trajectory
prediction using the Closed-form Continuous-time (CfC) cell, which is the modern,
solver-free evolution of the LTC architecture.

Mathematical Foundation:
-----------------------
The LTC architecture is based on the continuous differential equation (Eq. 1):

    dx(t)/dt = -[1/τ + f(x(t), I(t), t, θ)] x(t) + f(x(t), I(t), t, θ) A

where:
    - x(t): hidden state (context vector) at time t
    - τ: time constant (controls decay rate)
    - f(·): bounded, sigmoidal nonlinearity ensuring stability
    - I(t): input features (velocity, angular velocity)
    - A: learnable attractor state

The update step (Eq. 3) is computed as:

    x(t + Δt) = [x(t) + Δt·f(·)·A] / [1 + Δt·(1/τ + f(·))]

Key Guarantees (from Theorems 1 & 2):
-------------------------------------
1. Bounded Time-Constant: τᵢ/(1 + τᵢWᵢ) ≤ τₛᵧₛ,ᵢ ≤ τᵢ
    → Processing speed is controlled and stable

2. Bounded Hidden State: |x(t)| is bounded for all t
    → Network outputs never explode, even with unbounded inputs

Architectural Advantages:
------------------------
- Expressivity: Trajectory length 81.01 ± 10.05 vs CT-RNN's 4.05 ± 2.17
- Stability: Guaranteed bounds on hidden states prevent divergence
- Adaptivity: Dynamic time constants adjust to input characteristics
- Continuous: Natural modeling of physical motion dynamics

References:
----------
[1] Hasani et al., "Liquid Time-Constant Networks", AAAI 2021
[2] Hasani et al., "Neural Circuit Policies", NeurIPS 2020
[3] Hasani et al., "Closed-form Continuous-time Neural Networks", Nature ML 2022
"""

import torch
import torch.nn as nn
from ncps.torch import CfC  # Closed-form Continuous-time cell
import torch.utils.checkpoint as checkpoint


class LiquidVelocityModel(nn.Module):
    """
    Liquid Time-Constant Network for residual velocity prediction.

    Architecture Overview:
    ---------------------
    1. Encoder (CfC): Maps history sequence H → continuous context state h_enc
    2. Decoder (CfC): Autoregressively predicts residuals (Δv, Δω) from h_enc
    3. Integration: Combines base velocities + residuals → trajectory positions

    The model predicts corrections to average speed (v_avg) and angular velocity
    (omega_avg), which are then integrated using kinematic equations to produce
    position predictions.

    Key Differences from LSTM:
    -------------------------
    - LSTM: Discrete algebraic gates (h_n, c_n) with fixed update rules
    - LTC/CfC: Continuous differential equations with adaptive time constants

    Input Shape: (B, A, N, H, 2) where:
        B = batch size
        A = max agents (padded)
        N = 1 (ego agent) + num_neighbors
        H = history length (8 timesteps)
        2 = (x, y) coordinates

    Output Shape: (B, A, T, 2) where T = prediction_len (12 timesteps)
    """

    def __init__(
        self,
        input_size=2,
        hidden_size=64,
        dt=0.25,
        target_avg_speed=5.15,
        speed_tolerance=0.15,
        prediction_len=12,
        max_neighbors=4,
        backbone_layers=1,
        backbone_units=None,
        backbone_dropout=0.0,
        activation='hardtanh',
        dense_dt=0.05
    ):
        """
        Initialize the Liquid Velocity Model with social context.

        Args:
            input_size (int): Dimension of input features (default: 2 for x, y)
            hidden_size (int): Number of hidden units in CfC cells (context dimension)
            dt (float): Time step for integration (seconds)
            target_avg_speed (float): Expected average speed (units/s)
            speed_tolerance (float): Allowed speed deviation (fraction)
            prediction_len (int): Number of future timesteps to predict
            max_neighbors (int): Maximum number of neighbors to consider (default: 4)
            backbone_layers (int): Number of stacked CfC layers (1 recommended)
            backbone_units (int): Hidden units in internal CfC backbone (None = hidden_size)
            backbone_dropout (float): Dropout rate in CfC backbone
            activation (str): Activation function for social processor ('hardtanh', 'relu', 'tanh')
            dense_dt (float): Time step for dense trajectory integration (seconds, default: 0.05)
        """
        super().__init__()

        # Hyperparameters
        self.hidden_size = hidden_size
        self.dt = dt
        self.dense_dt = dense_dt
        self.target_avg_speed = target_avg_speed
        self.speed_tolerance = speed_tolerance
        self.prediction_len = prediction_len
        self.max_neighbors = max_neighbors

        # Speed clamping bounds (prevent unrealistic velocities)
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)

        # ========================================================================
        # SOCIAL CONTEXT PROCESSOR (IMPROVEMENT 1)
        # ========================================================================
        # Aggregates multi-agent features: ego + neighbors → hidden representation
        # Input: (B*A, H, N*2) where N = 1 (ego) + max_neighbors
        # Output: (B*A, H, hidden_size)

        # Select activation function based on LTC expressivity research
        # Paper shows Hard-tanh achieves 81.01 trajectory length (highest)
        if activation == 'hardtanh':
            act_fn = nn.Hardtanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.Tanh()

        # Social context aggregation network
        social_input_dim = (1 + max_neighbors) * 2  # All agents' (x, y) positions
        self.social_processor = nn.Sequential(
            nn.Linear(social_input_dim, hidden_size * 2),
            act_fn,
            nn.Dropout(backbone_dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            act_fn,
            nn.LayerNorm(hidden_size)  # Stabilize features before CfC
        )

        # Legacy input embedding (for backward compatibility with ego-only mode)
        self.input_embed = nn.Linear(input_size, hidden_size)

        # ========================================================================
        # ENCODER: CfC cell for history encoding with social context
        # ========================================================================
        # Maps social-aware history sequence (B*A, H, hidden_size) → context vector (B*A, hidden_size)
        # Input features I(t) now include ego + neighbor positions at each timestep
        # The CfC cell internally solves the continuous dynamics (Eq. 1) using
        # the closed-form update (Eq. 3) at each timestep.
        self.encoder = CfC(
            input_size=hidden_size,
            units=hidden_size,
            proj_size=None,  # No projection, direct hidden state output
            return_sequences=False,  # Only return final hidden state
            batch_first=True,
            mixed_memory=True,  # Enables both short and long-term dependencies
            mode="default",  # Use standard CfC dynamics
            backbone_layers=backbone_layers,
            backbone_units=backbone_units if backbone_units else hidden_size,
            backbone_dropout=backbone_dropout
        )

        # ========================================================================
        # DECODER: CfC cell for autoregressive prediction
        # ========================================================================
        # At each prediction step t, takes:
        #   - Input: embedded residual from step t-1 (B*A, 1, hidden_size)
        #   - Hidden: continuous state h_x from step t-1 (B*A, hidden_size)
        # Outputs:
        #   - Updated hidden state h_x' for step t
        #   - Output features for predicting (Δv, Δω)
        self.decoder_rnn = CfC(
            input_size=hidden_size,
            units=hidden_size,
            proj_size=None,
            return_sequences=True,  # Return output at each step
            batch_first=True,
            mixed_memory=True,
            mode="default",
            backbone_layers=backbone_layers,
            backbone_units=backbone_units if backbone_units else hidden_size,
            backbone_dropout=backbone_dropout
        )

        # Output head: hidden state → (Δv, Δω) residuals
        self.output = nn.Linear(hidden_size, 2)

        # Optional post-processing layer (can be used for additional refinement)
        self.postprocess = nn.Linear(2, 2)

    def compute_velocity_features(self, history):
        """
        Extract kinematic features from position history.

        Computes:
        1. v_avg: Average speed over history (magnitude of velocity)
        2. omega_avg: Average angular velocity (HARD-CODED TO ZERO)
        3. theta_last: Heading angle at last timestep

        Mathematical Details:
        --------------------
        For positions p_t, p_{t+1}:
            displacement = p_{t+1} - p_t
            distance = ||displacement||₂
            velocity = distance / dt
            heading θ = atan2(dy, dx)

        Args:
            history: Tensor (B, A, H, 2) - position history

        Returns:
            v_avg: Tensor (B, A) - average speeds
            omega_avg: Tensor (B, A) - angular velocities (zeros)
            theta_last: Tensor (B, A) - final heading angles
        """
        B, A, H, _ = history.shape
        device = history.device

        # Compute displacements: Δp = p_{t+1} - p_t
        displacements = history[:, :, 1:, :] - history[:, :, :-1, :]  # (B, A, H-1, 2)

        # Compute distances and velocities
        distances = torch.norm(displacements, dim=-1)  # (B, A, H-1)
        velocities = distances / self.dt  # (B, A, H-1)

        # Compute heading angles
        dx = displacements[:, :, :, 0]  # (B, A, H-1)
        dy = displacements[:, :, :, 1]  # (B, A, H-1)
        theta = torch.atan2(dy, dx)  # (B, A, H-1)

        # DESIGN CHOICE: Force omega_avg to zero (straight-line assumption)
        # This simplifies the dynamics and focuses learning on speed variations
        omega_avg = torch.zeros(B, A, device=device)

        # Compute average velocity (with masking for very low velocities)
        valid_mask_v = velocities > 0.1  # Filter noise/stationary periods
        v_avg = torch.where(
            valid_mask_v.any(dim=2),
            velocities.sum(dim=2) / (valid_mask_v.sum(dim=2).float() + 1e-6),
            torch.full((B, A), self.target_avg_speed, device=device)
        )

        # Extract last heading (or zero if history too short)
        theta_last = theta[:, :, -1] if H > 1 else torch.zeros(B, A, device=device)

        return v_avg, omega_avg, theta_last

    def integrate_velocities_to_positions(
        self,
        velocity_residuals,
        v_avg,
        omega_avg,
        theta_last,
        last_pos
    ):
        """
        Integrate velocity predictions into position trajectory.

        Kinematic Integration:
        ---------------------
        At each timestep t:
            1. v_pred = v_avg + Δv_t (apply residual to base speed)
            2. ω_pred = ω_avg + Δω_t (apply residual to angular velocity)
            3. v_pred = clamp(v_pred, min_speed, max_speed)
            4. θ_t = θ_{t-1} + ω_pred * dt (update heading)
            5. θ_t = atan2(sin(θ_t), cos(θ_t)) (normalize to [-π, π])
            6. dx = v_pred * cos(θ_t) * dt
            7. dy = v_pred * sin(θ_t) * dt
            8. p_t = p_{t-1} + (dx, dy)

        Args:
            velocity_residuals: Tensor (B, A, T, 2) - predicted (Δv, Δω)
            v_avg: Tensor (B, A) - base speeds
            omega_avg: Tensor (B, A) - base angular velocities
            theta_last: Tensor (B, A) - initial headings
            last_pos: Tensor (B, A, 2) - starting positions

        Returns:
            positions: Tensor (B, A, T, 2) - predicted trajectory
        """
        B, A, T, _ = velocity_residuals.shape
        positions = torch.zeros_like(velocity_residuals)

        # Initialize integration state
        current_pos = last_pos.clone()
        current_theta = theta_last.clone()

        # Iterative integration over prediction horizon
        for t in range(T):
            # Extract residuals for current timestep
            delta_v = velocity_residuals[:, :, t, 0]      # (B, A)
            delta_omega = velocity_residuals[:, :, t, 1]  # (B, A)

            # Apply residuals to base velocities
            v_pred = v_avg + delta_v
            omega_pred = omega_avg + delta_omega  # omega_avg is always zero

            # Clamp speed to realistic bounds (prevent instability)
            v_pred = torch.clamp(v_pred, self.min_speed, self.max_speed)

            # Update heading with angular velocity
            current_theta = current_theta + omega_pred * self.dt

            # Normalize angle to [-π, π] (prevents accumulation of error)
            current_theta = torch.atan2(
                torch.sin(current_theta),
                torch.cos(current_theta)
            )

            # Compute displacement in Cartesian coordinates
            dx = v_pred * torch.cos(current_theta) * self.dt
            dy = v_pred * torch.sin(current_theta) * self.dt

            # Update position
            current_pos = current_pos + torch.stack([dx, dy], dim=-1)
            positions[:, :, t] = current_pos

        return positions

    def forward(self, history_neighbors, expanded_goals=None, neighbors=None):
        """
        Forward pass: Social History → Liquid Dynamics → Residuals → Fixed-Step Positions

        Standard Pipeline (Fixed Timesteps):
        -----------------------------------
        1. Aggregate ego + neighbor history → social input features I(t)
        2. Process social features → CfC-compatible hidden representation
        3. ENCODER: CfC encoding with social context → context vector h_enc
        4. DECODER: Autoregressively predict T residuals using liquid dynamics
        5. INTEGRATION: Fixed-step integration at resolution dt (0.25s)

        Mathematical Flow (Per Agent with Social Context):
        -------------------------------------------------
        Input: H_social ∈ ℝ^(H×N×2) (ego + N neighbors' positions)

        Social Processing:
            H_flat = Flatten(H_social) ∈ ℝ^(H×(N*2))
            I(t) = SocialProcessor(H_flat) ∈ ℝ^(H×U)  [Aggregate features]

        Encoder:
            h_enc = CfC_encode(I(t)) ∈ ℝ^U  [Eq. 1 solved over H steps]

        Decoder (for t = 1..T):
            h_t, out_t = CfC_decode(r_{t-1}, h_{t-1})  [Eq. 1 solved for 1 step)
            r_t = Linear(out_t) ∈ ℝ^2  [Δv_t, Δω_t]

        Fixed-Step Integration (for t = 1..T):
            v_t = v_avg + Δv_t
            ω_t = 0 + Δω_t
            θ_t = θ_{t-1} + ω_t·dt
            p_t = p_{t-1} + v_t·[cos(θ_t), sin(θ_t)]·dt

        Args:
            history_neighbors: Tensor (B, A, N, H, 2) - history with neighbors
                B = batch size
                A = max agents (padded)
                N = 1 (ego) + num_neighbors (up to max_neighbors)
                H = history length
                2 = (x, y) coordinates
            expanded_goals: Optional, not used in this model
            neighbors: Optional, not used in this model

        Returns:
            positions: Tensor (B, A, T, 2) - predicted trajectories (at dt resolution)
            velocity_residuals: Tensor (B, A, T, 2) - predicted (Δv, Δω) at dt resolution
            None: Placeholder for compatibility
        """
        # ====================================================================
        # 1. INPUT PREPARATION AND SOCIAL CONTEXT AGGREGATION
        # ====================================================================

        # Input shape: (B, A, N, H, 2) or (B, A, H, 2)
        if history_neighbors.dim() == 5:
            B, A, N, H, D = history_neighbors.shape
        elif history_neighbors.dim() == 4:
            # If no neighbor dimension, add it
            B, A, H, D = history_neighbors.shape
            history_neighbors = history_neighbors.unsqueeze(2)  # Add N dimension
            N = 1
        else:
            raise ValueError(f"Expected history_neighbors to be 4D or 5D, got {history_neighbors.dim()}D with shape {history_neighbors.shape}")
        
        device = history_neighbors.device

        # Extract ego agent history for kinematic feature computation
        history_ego = history_neighbors[:, :, 0, :, :]  # (B, A, H, 2)

        # Compute deterministic base velocity features from ego history
        v_avg, omega_avg, theta_last = self.compute_velocity_features(history_ego)
        # v_avg: (B, A) - average speeds
        # omega_avg: (B, A) - zeros
        # theta_last: (B, A) - final heading angles

        # ====================================================================
        # SOCIAL CONTEXT PROCESSING (IMPROVEMENT 1)
        # ====================================================================
        # Aggregate multi-agent features: flatten N agents' positions per timestep
        # Shape transformation: (B, A, N, H, 2) → (B, A, H, N*2)

        # Pad neighbors dimension if necessary (handle variable number of neighbors)
        if N < (1 + self.max_neighbors):
            pad_size = (1 + self.max_neighbors) - N
            # Pad with zeros: (B, A, pad_size, H, 2)
            padding = torch.zeros(B, A, pad_size, H, D, device=device)
            history_neighbors_padded = torch.cat([history_neighbors, padding], dim=2)
        else:
            # Truncate if too many neighbors
            history_neighbors_padded = history_neighbors[:, :, :(1 + self.max_neighbors), :, :]

        # Flatten agent dimension per timestep: (B, A, H, (1+max_neighbors)*2)
        social_input = history_neighbors_padded.permute(0, 1, 3, 2, 4)  # (B, A, H, N, 2)
        social_input = social_input.reshape(B, A, H, -1)  # (B, A, H, N*2)

        # Flatten batch and agent dimensions for processing
        social_input_flat = social_input.reshape(B * A, H, -1)  # (B*A, H, N*2)

        # ====================================================================
        # 2. ENCODER PASS: Social Context → Continuous Context State
        # ====================================================================

        # Process social input through aggregation network
        # Input I(t): (B*A, H, N*2) → (B*A, H, hidden_size)
        social_features = self.social_processor(social_input_flat)

        # CfC ENCODER: Solve continuous dynamics over socially-aware history
        # Returns: (sequence_output, final_hidden_state)
        # We only need the final hidden state (context vector)
        _, h_n_ctx = self.encoder(social_features)  # h_n_ctx: (B*A, hidden_size)

        # ====================================================================
        # 3. DECODER PASS: Autoregressive Residual Prediction
        # ====================================================================

        # Initialize decoder state with encoder context
        hx = h_n_ctx  # (B*A, hidden_size)

        T = self.prediction_len
        velocity_residuals = []

        # Start with zero residual input (cold start)
        current_input = torch.zeros(B * A, 1, D, device=device)  # (B*A, 1, 2)

        # Autoregressive loop: predict T timesteps sequentially
        for t in range(T):
            # Embed current input residual
            input_embedded = self.input_embed(current_input)  # (B*A, 1, hidden_size)

            # CfC DECODER STEP: Solve continuous dynamics for one timestep
            # Input: embedded residual + previous hidden state
            # Output: updated hidden state + output features
            out, hx = self.decoder_rnn(input_embedded, hx)
            # out: (B*A, 1, hidden_size)
            # hx: (B*A, hidden_size) - updated continuous state

            # Predict residual (Δv, Δω) from output features
            step_output = self.output(out.squeeze(1))  # (B*A, 2)
            step_output = self.postprocess(step_output)  # Optional refinement

            # Store prediction
            velocity_residuals.append(step_output.view(B * A, 1, 2))

            # AUTOREGRESSIVE FEEDBACK: Use predicted residual as next input
            current_input = step_output.unsqueeze(1)  # (B*A, 1, 2)

        # Concatenate all predicted residuals
        velocity_residuals = torch.cat(velocity_residuals, dim=1)  # (B*A, T, 2)
        velocity_residuals = velocity_residuals.view(B, A, T, 2)

        # ====================================================================
        # 4. FIXED-STEP KINEMATIC INTEGRATION: Residuals → Fixed-Step Positions
        # ====================================================================

        # Generate fixed-step trajectory at the self.dt resolution (0.25s)
        # The model reverts to the standard output format.
        positions = self.integrate_velocities_to_positions(
            velocity_residuals,
            v_avg,
            omega_avg,
            theta_last,
            history_ego[:, :, -1, :]  # Last position from ego history
        )

        return positions, velocity_residuals, None


# Alias for backward compatibility
DeepLiquidVelocityModel = LiquidVelocityModel


# ============================================================================
# ARCHITECTURE SUMMARY
# ============================================================================
"""
Model: LiquidVelocityModel (Liquid Time-Constant Network)
---------------------------------------------------------

Theoretical Foundation:
    - Continuous differential equation: dx/dt = -[1/τ + f(·)]x + f(·)A
    - Closed-form solver: x(t+Δt) = [x(t) + Δt·f(·)·A] / [1 + Δt·(1/τ + f(·))]
    - Guaranteed stability: |x(t)| bounded, τ_sys bounded

Key Components (ENHANCED):
    1. Social Context Processor: Multi-agent (x,y) → aggregated features
    2. CfC Encoder: Social-aware history → h_enc (continuous context vector)
    3. CfC Decoder: h_enc → (Δv, Δω) residuals (autoregressive)
    4. Output Head: hidden → (Δv, Δω)
    5. Fixed-Step Kinematic Integrator: (v_avg, Δv, Δω) → standard resolution trajectory

Enhancements over Base LTC:
    - Social Context: Processes ego + neighbor positions (max_neighbors)
    - Activation: Hardtanh/ReLU for maximum expressivity (81.01 trajectory length)
    - Layer Normalization: Stabilizes social features before CfC processing
    - Output Resolution: Fixed to dt (e.g., 0.25s)

Advantages over LSTM:
    - Higher expressivity (81.01 vs 4.05 trajectory length)
    - Guaranteed stability (no exploding gradients/states)
    - Adaptive time constants (better captures dynamics)
    - Continuous modeling (natural for physical systems)
    - Social awareness (neighbor-conditioned predictions)

Training Considerations:
    - Use MSE loss on positions (or ADE/FDE metrics)
    - Learning rate: ~1e-4 (CfC is stable but can be sensitive)
    - Gradient clipping: Optional (stability guarantees help)
    - Batch size: Same as LSTM baseline
    - Social context improves accuracy but increases memory usage

Hyperparameters:
    - hidden_size: 64 (default) - balance expressivity/efficiency
    - dt: 0.25s - prediction timestep (must match data)
    - prediction_len: 12 - 3 seconds into future at dt resolution
    - max_neighbors: 4 - number of neighbors to consider
    - target_avg_speed: 5.15 - dataset-specific (pedestrian: ~1.4 m/s)
    - activation: 'hardtanh' - maximizes LTC expressivity
    - backbone_layers: 1 - deeper may overfit on small datasets
"""