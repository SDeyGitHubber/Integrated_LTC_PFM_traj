"""
================================================================================
LTC-PFM HYBRID MODEL: Liquid Time-Constant Network with Physics-Informed Motion
================================================================================

This module implements the hybrid trajectory prediction architecture that combines:
1. Liquid Time-Constant Networks (LTC/CfC) for neural dynamics
2. Potential Field Methods (PFM) for physics-informed motion priors

Based on the architecture from CheckpointedIntegratedMTAPFM_neighbours but
replacing LSTM encoder/decoder with tweakable CfC/LTC cells.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

Input: Historical positions H_ego, H_neighbors, Goal g
Initialize: x_0 ← last(H_ego), θ_0 ← θ_last, v_avg, ω_avg

Step 1: Embed history sequences
Step 2: h_enc ← CfC_Encoder(embedded_history)  [Social context encoding]
Step 3: coeffs ← CoefficientProjector(h_enc)   [Dynamic PFM coefficients]

For t = 1 to T_pred (Autoregressive Decoding Loop):
    Step 4:  decoded_pred ← CfC_Decoder_step(input_t, hx)    [Raw neural prediction]
    Step 5:  A_PFM ← PFM(x_{t-1}, decoded_pred, neighbors, g, coeffs)  [PFM forces]
    Step 6:  adjusted_pos ← x_{t-1} + A_PFM                   [Apply physics correction]
    Step 7:  Apply speed constraints to adjusted_pos
    Step 8:  Update kinematic state (θ, v, ω)

Return: adjusted_preds, decoded_preds, coefficients

================================================================================
KEY DIFFERENCES FROM LSTM VERSION
================================================================================

Original CheckpointedIntegratedMTAPFM_neighbours:
- LSTM Encoder processes history → (h_n, c_n)
- LSTM Decoder autoregressively predicts displacements
- PFM corrects predictions post-hoc

This LTC-PFM Hybrid Model:
- CfC/LTC Encoder with tweakable ODE dynamics
- CfC/LTC Decoder with continuous-time adaptation
- Angular velocity integration (like lstm_ang_vel_own)
- Full access to internal ODE equations for modification

================================================================================
MATHEMATICAL FOUNDATION
================================================================================

CfC Cell ODE (from LTC paper):
    dx(t)/dt = -[1/τ + f(x,I,θ)] x(t) + f(x,I,θ) A

Closed-form update:
    x(t+Δt) = [x(t) + Δt·f(·)·A] / [1 + Δt·(1/τ + f(·))]

PFM Forces:
    F_goal = k1 · (goal - pos)           [Goal attraction]
    F_pred = k2 · (predicted - pos)      [Prediction attraction]  
    F_rep = Σ kr · (pos - nbr) / ||d||²  [Neighbor repulsion]

Kinematic Integration:
    v_t = clamp(v_avg + Δv, v_min, v_max)
    θ_t = θ_{t-1} + ω_t · Δt
    x_t = x_{t-1} + v_t · cos(θ_t) · Δt
    y_t = y_{t-1} + v_t · sin(θ_t) · Δt

================================================================================
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import sys
import os

# Add parent directory to path for utils import (safe for notebooks)
# In notebook environments like Kaggle, __file__ may not be defined.
try:
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback: use current working directory when __file__ is unavailable
    _BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

if _BASE_DIR not in sys.path:
    sys.path.append(_BASE_DIR)

from utils.cfc_cell import CfCCell, CfC


# =============================================================================
# POTENTIAL FIELD MODULE
# =============================================================================

class PotentialField(nn.Module):
    """
    Physics-inspired module that computes social forces on each agent.
    
    This module models:
    - Attraction to goals (F_goal)
    - Repulsion from neighbors (F_rep)
    - Attraction to neural prediction (F_pred)
    
    Following principles of potential field methods for collision-free 
    and goal-directed path planning.
    
    Mathematical Formulation:
    -------------------------
    F_goal = k1 · (goal - pos)                    [Attractive to goal]
    F_pred = k2 · (predicted - pos)               [Attractive to prediction]
    F_rep = Σ_j [kr · (pos - nbr_j) / ||d||²]    [Repulsive from neighbors]
    
    Total: A_PFM = F_goal + F_pred + F_rep
    
    Args:
        num_agents (int): Number of unique agents (for coefficient embedding)
        k_init (float): Initial value for all force coefficients
        repulsion_radius (float): Maximum distance for neighbor repulsion
    """

    def __init__(
        self,
        num_agents: int = 1000,
        k_init: float = 1.0,
        repulsion_radius: float = 0.5
    ):
        super().__init__()
        self.repulsion_radius = repulsion_radius
        
        # Learnable coefficients per agent: [k_goal, k_pred, k_rep]
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)

    def forward(
        self,
        pos: torch.Tensor,
        predicted: torch.Tensor,
        neighbors: torch.Tensor,
        goal: torch.Tensor,
        coeffs: torch.Tensor
    ) -> tuple:
        """
        Compute total potential field force for each agent.

        Args:
            pos: Current positions [B, A, D]
            predicted: Raw predicted positions [B, A, D] or [B, A, 1, D]
            neighbors: Neighbor positions [B, A, N, D]
            goal: Goal positions [B, A, D]
            coeffs: Force coefficients [B, A, 3]

        Returns:
            total_force: Net force [B, A, D]
            coeffs: Same coefficients (for tracking/logging)
        """
        # Extract coefficients
        k1 = coeffs[..., 0:1]  # Goal attraction
        k2 = coeffs[..., 1:2]  # Prediction attraction
        kr = coeffs[..., 2:3]  # Neighbor repulsion

        # =================================================================
        # FORCE 1: Goal Attraction
        # F_goal = k1 · (goal - pos)
        # =================================================================
        F_goal = k1 * (goal - pos)

        # =================================================================
        # FORCE 2: Prediction Attraction
        # F_pred = k2 · (predicted - pos)
        # =================================================================
        if predicted.dim() == 3:
            F_pred = k2 * (predicted - pos)
        else:
            # Handle [B, A, 1, D] shape
            F_pred = k2 * (predicted[:, :, 0, :] - pos)

        # =================================================================
        # FORCE 3: Neighbor Repulsion
        # F_rep = Σ_j [kr · (pos - nbr_j) / ||pos - nbr_j||²] · mask
        # =================================================================
        if neighbors.size(2) == 0:
            # No neighbors - zero repulsion
            F_rep = torch.zeros_like(pos)
        else:
            # Compute differences: pos - each neighbor
            diffs = pos.unsqueeze(2) - neighbors  # [B, A, N, D]
            dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6  # [B, A, N, 1]
            
            # Apply repulsion only within radius and for non-zero distances
            mask = (dists < self.repulsion_radius) & (dists > 1e-5)
            mask = mask.float()

            # Expand kr for broadcasting
            kr_exp = kr.unsqueeze(2)  # [B, A, 1, 1]
            safe_dists = torch.max(dists, torch.tensor(1e-6, device=dists.device))

            # Repulsion inversely proportional to distance squared
            repulsion = kr_exp * diffs / safe_dists.pow(2) * mask
            F_rep = repulsion.sum(dim=2)  # Sum over neighbors

        # Total force
        total_force = F_goal + F_pred + F_rep
        
        return total_force, coeffs


# =============================================================================
# MAIN HYBRID MODEL
# =============================================================================

class LTC_PFM_HybridModel(nn.Module):
    """
    Hybrid Neural/Physics Model for Multi-Agent Trajectory Prediction.
    
    Combines CfC/LTC encoder-decoder with Physics-Informed Potential Fields.
    
    Architecture Flow:
    -----------------
    1. Input Processing:
       - Embed history positions through linear layer
       - Flatten batch and agent dimensions
    
    2. CfC Encoder:
       - Process embedded history with continuous-time dynamics
       - Output: encoded context h_enc
    
    3. Coefficient Projection:
       - Project h_enc to PFM coefficients [k_goal, k_pred, k_rep]
    
    4. Autoregressive Decoding Loop (for t = 1..T):
       a. CfC Decoder step: predict raw displacement
       b. Accumulate to get decoded_preds
       c. PFM: compute physics forces from current state
       d. Apply force correction to get adjusted_preds
       e. Apply speed constraints
       f. Update kinematic state (optional angular velocity)
    
    5. Output:
       - adjusted_preds: Physics-corrected trajectory
       - decoded_preds: Raw neural predictions
       - coeff_mean, coeff_var: Coefficient statistics
    
    Args:
        input_size: Dimension of positions (default: 2 for x,y)
        hidden_size: CfC hidden state dimension
        num_layers: Number of stacked CfC layers (via backbone)
        target_avg_speed: Expected average speed
        speed_tolerance: Allowed speed deviation fraction
        num_agents: Max agents for coefficient embedding
        dt: Integration timestep
        cfc_mode: CfC cell mode ("default", "pure", "no_gate")
        cfc_backbone_units: Units in CfC backbone MLP
        cfc_backbone_layers: Layers in CfC backbone MLP
        cfc_backbone_activation: Activation function for backbone
        use_angular_velocity: If True, integrate with angular velocity
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,  # Used for backbone_layers
        target_avg_speed: float = 4.087,
        speed_tolerance: float = 0.15,
        num_agents: int = 1000,
        dt: float = 0.1,
        # CfC-specific parameters (TWEAKABLE)
        cfc_mode: str = "default",
        cfc_backbone_units: int = 128,
        cfc_backbone_layers: int = 1,
        cfc_backbone_activation: str = "lecun_tanh",
        cfc_backbone_dropout: float = 0.0,
        mixed_memory: bool = True,
        # Angular velocity integration
        use_angular_velocity: bool = True
    ):
        super().__init__()
        
        # Store hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dt = dt
        self.use_angular_velocity = use_angular_velocity
        
        # Speed constraints
        if target_avg_speed is None:
            raise ValueError("target_avg_speed must be provided")
        self.target_avg_speed = target_avg_speed
        self.speed_tolerance = speed_tolerance
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)
        
        # =================================================================
        # INPUT EMBEDDING
        # Maps raw positions to hidden dimension
        # =================================================================
        self.input_embed = nn.Linear(input_size, hidden_size)
        
        # =================================================================
        # CfC ENCODER (Replaces LSTM Encoder)
        # =================================================================
        # Uses our tweakable CfC implementation from utils
        # 
        # Mathematical Dynamics:
        # dx/dt = -[1/τ + f(x,I,θ)] x + f(x,I,θ) A
        # 
        # The encoder processes the embedded history sequence and
        # outputs a context vector h_enc that captures temporal patterns.
        # =================================================================
        self.encoder = CfC(
            input_size=hidden_size,
            units=hidden_size,
            proj_size=None,
            return_sequences=False,  # Only final hidden state
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=cfc_mode,
            activation=cfc_backbone_activation,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            backbone_dropout=cfc_backbone_dropout
        )
        
        # =================================================================
        # CfC DECODER (Replaces LSTM Decoder)
        # =================================================================
        # Autoregressive decoder that predicts residual displacements
        # at each timestep, conditioned on encoder output.
        # =================================================================
        self.decoder = CfC(
            input_size=hidden_size,
            units=hidden_size,
            proj_size=None,
            return_sequences=True,  # Return output at each step
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=cfc_mode,
            activation=cfc_backbone_activation,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            backbone_dropout=cfc_backbone_dropout
        )
        
        # =================================================================
        # COEFFICIENT PROJECTOR
        # Maps encoder hidden state to PFM coefficients
        # =================================================================
        self.coeff_projector = nn.Linear(hidden_size, 3)
        
        # =================================================================
        # OUTPUT HEAD
        # Maps decoder hidden state to position residuals
        # If use_angular_velocity: output (Δx, Δy, Δω) → then we use (Δv, Δω)
        # =================================================================
        if use_angular_velocity:
            self.output = nn.Linear(hidden_size, 2)  # (Δv, Δω) residuals
        else:
            self.output = nn.Linear(hidden_size, input_size)  # (Δx, Δy) residuals
        
        # =================================================================
        # POTENTIAL FIELD MODULE
        # =================================================================
        self.pfm = PotentialField(
            num_agents=num_agents,
            k_init=1.0,
            repulsion_radius=0.5
        )
    
    def compute_velocity_features(self, history: torch.Tensor) -> tuple:
        """
        Extract kinematic features from position history.
        
        Computes:
        - v_avg: Average speed over each agent's own history
        - omega_avg: Average angular velocity (set to 0 for stability)
        - theta_last: Final heading angle
        
        Args:
            history: [B, A, H, 2] position history
            
        Returns:
            v_avg: [B, A] average speeds
            omega_avg: [B, A] angular velocities (zeros)
            theta_last: [B, A] final heading angles
        """
        B, A, H, _ = history.shape
        device = history.device
        
        # Compute displacements
        displacements = history[:, :, 1:, :] - history[:, :, :-1, :]  # [B, A, H-1, 2]
        
        # Compute velocities
        distances = torch.norm(displacements, dim=-1)  # [B, A, H-1]
        velocities = distances / self.dt
        
        # Compute headings
        dx = displacements[:, :, :, 0]
        dy = displacements[:, :, :, 1]
        theta = torch.atan2(dy, dx)  # [B, A, H-1]
        
        # Force omega_avg to zero (straight-line assumption)
        omega_avg = torch.zeros(B, A, device=device)
        
        # Average velocity with masking, per agent history
        valid_mask = velocities > 0.1
        valid_velocities = velocities * valid_mask.float()
        v_avg = torch.where(
            valid_mask.any(dim=2),
            valid_velocities.sum(dim=2) / (valid_mask.sum(dim=2).float() + 1e-6),
            torch.zeros(B, A, device=device)
        )
        
        # Last heading
        theta_last = theta[:, :, -1] if H > 1 else torch.zeros(B, A, device=device)
        
        return v_avg, omega_avg, theta_last
    
    def forward_encoder(self, x: torch.Tensor) -> tuple:
        """
        CfC encoder forward pass (for checkpointing).
        
        Args:
            x: Embedded history [B*A*ent, H, hidden_size]
            
        Returns:
            output: Sequence output (not used)
            hx: Final hidden state
        """
        return self.encoder(x)
    
    def forward_decoder_step(
        self,
        input_step: torch.Tensor,
        hx: torch.Tensor
    ) -> tuple:
        """
        Single CfC decoder step (for checkpointing).
        
        Args:
            input_step: Input at current step [B*A*ent, 1, hidden_size]
            hx: Previous hidden state
            
        Returns:
            output: Decoder output
            hx: Updated hidden state
        """
        return self.decoder(input_step, hx)
    
    def forward(
        self,
        history_neighbors: torch.Tensor,
        goal: torch.Tensor
    ) -> tuple:
        """
        Main forward pass for trajectory prediction with PFM correction.
        
        Args:
            history_neighbors: [B, A, ent, H, 2]
                - B: batch size
                - A: number of agents
                - ent: entities (1=ego + neighbors)
                - H: history length
                - 2: (x, y) coordinates
            goal: [B, A, ent, 2] or [B, A, 2]
                Goal positions
        
        Returns:
            adjusted_preds: [B, A, ent, T, 2] - Physics-corrected predictions
            decoded_preds: [B, A, ent, T, 2] - Raw neural predictions
            coeff_mean: Scalar - Mean of coefficients
            coeff_var: Scalar - Variance of coefficients
        """
        # Extract dimensions
        B, A, ent, H, D = history_neighbors.shape
        device = history_neighbors.device
        
        # =================================================================
        # GOAL SHAPE HANDLING
        # =================================================================
        if goal.shape == (B, A, D):
            goal = goal.unsqueeze(2).expand(B, A, ent, D).contiguous()
        elif goal.shape != (B, A, ent, D):
            raise AssertionError(
                f"Goal shape mismatch: got {goal.shape}, expected {(B, A, D)} or {(B, A, ent, D)}"
            )
        
        # =================================================================
        # INPUT EMBEDDING
        # =================================================================
        hist_flat = history_neighbors.reshape(B * A * ent, H, D)
        emb = self.input_embed(hist_flat)  # [B*A*ent, H, hidden_size]
        
        # =================================================================
        # CfC ENCODER
        # =================================================================
        # Uses gradient checkpointing for memory efficiency
        _, h_enc = cp.checkpoint(self.forward_encoder, emb, use_reentrant=False)
        # h_enc shape depends on mixed_memory:
        # - If mixed_memory=True: (h_state, c_state) tuple
        # - If mixed_memory=False: h_state tensor
        
        if isinstance(h_enc, tuple):
            h_state, c_state = h_enc
            hx = (h_state, c_state)
            h_for_proj = h_state
        else:
            hx = h_enc
            h_for_proj = h_enc
        
        # =================================================================
        # COEFFICIENT PROJECTION
        # =================================================================
        h_top = h_for_proj.reshape(B, A, ent, self.hidden_size)
        coeffs = self.coeff_projector(h_top)  # [B, A, ent, 3]
        
        # =================================================================
        # COMPUTE KINEMATIC FEATURES (if using angular velocity)
        # =================================================================
        if self.use_angular_velocity:
            # Extract ego history for kinematics
            history_ego = history_neighbors[:, :, 0, :, :]  # [B, A, H, 2]
            v_avg, omega_avg, theta_last = self.compute_velocity_features(history_ego)
            # These are [B, A] tensors
            v_avg_exp = v_avg.view(B * A).unsqueeze(1).expand(B * A, ent).reshape(B * A * ent)
            v_min_exp = v_avg_exp * (1 - self.speed_tolerance)
            v_max_exp = v_avg_exp * (1 + self.speed_tolerance)
        
        # =================================================================
        # INITIALIZE PREDICTION STATE
        # =================================================================
        pred_len = 12
        pred_flat = torch.zeros(B * A * ent, pred_len, D, device=device)
        
        # Find last valid position for each entity
        history_flat = history_neighbors.view(B * A * ent, H, D)
        last_pos = torch.zeros(B * A * ent, 1, D, device=device)
        
        for idx in range(B * A * ent):
            for t in range(H - 1, -1, -1):
                pos = history_flat[idx, t]
                if not torch.all(pos == 0):
                    last_pos[idx] = pos
                    break
        
        # Initialize kinematic state for angular velocity mode
        if self.use_angular_velocity:
            current_theta = theta_last.view(B * A).clone()
            # Expand for all entities
            current_theta_ent = current_theta.unsqueeze(1).expand(B * A, ent).reshape(B * A * ent)
        
        # =================================================================
        # AUTOREGRESSIVE DECODING LOOP
        # =================================================================
        for t in range(pred_len):
            # Prepare decoder input
            if t == 0:
                decoder_in = last_pos.clone()
            else:
                decoder_in = pred_flat[:, t - 1:t].clone()
            
            # Embed decoder input
            dec_emb = self.input_embed(decoder_in)  # [B*A*ent, 1, hidden]
            
            # Ensure hx is contiguous for checkpointing
            if isinstance(hx, tuple):
                hx = (hx[0].contiguous(), hx[1].contiguous())
            else:
                hx = hx.contiguous()
            
            # CfC decoder step
            out, hx = cp.checkpoint(
                self.forward_decoder_step, dec_emb, hx, use_reentrant=False
            )
            
            # Output projection
            step_out = self.output(out.squeeze(1))  # [B*A*ent, output_dim]
            
            # =============================================================
            # KINEMATIC INTEGRATION (if using angular velocity)
            # =============================================================
            if self.use_angular_velocity:
                # step_out contains (Δv, Δω)
                delta_v = step_out[:, 0]      # [B*A*ent]
                delta_omega = step_out[:, 1]  # [B*A*ent]
                
                # Expand omega_avg to all entities
                omega_avg_exp = omega_avg.view(B * A).unsqueeze(1).expand(B * A, ent).reshape(B * A * ent)
                
                # Compute predicted velocity and angular velocity
                v_pred = v_avg_exp + delta_v
                omega_pred = omega_avg_exp + delta_omega
                
                # Clamp speed around each agent's own history speed
                v_pred = torch.clamp(v_pred, v_min_exp, v_max_exp)
                
                # Update heading
                current_theta_ent = current_theta_ent + omega_pred * self.dt
                current_theta_ent = torch.atan2(
                    torch.sin(current_theta_ent),
                    torch.cos(current_theta_ent)
                )
                
                # Compute displacement from velocity and heading
                dx = v_pred * torch.cos(current_theta_ent) * self.dt
                dy = v_pred * torch.sin(current_theta_ent) * self.dt
                displacement = torch.stack([dx, dy], dim=-1)  # [B*A*ent, 2]
                
                # Accumulate position
                if t == 0:
                    pred_flat[:, t] = last_pos.squeeze(1) + displacement
                else:
                    pred_flat[:, t] = pred_flat[:, t - 1] + displacement
            else:
                # Direct displacement prediction
                if t == 0:
                    pred_flat[:, t] = last_pos.squeeze(1) + step_out
                else:
                    pred_flat[:, t] = pred_flat[:, t - 1] + step_out
        
        # Reshape decoded predictions
        decoded_preds = pred_flat.reshape(B, A, ent, pred_len, D)
        
        # =================================================================
        # PFM CORRECTION LOOP
        # =================================================================
        adjusted_preds = torch.zeros_like(decoded_preds)
        current_pos = last_pos.clone().reshape(B, A, ent, D)
        
        coeff_list = []
        for t in range(pred_len):
            # Process each entity
            for idx in range(ent):
                pos = current_pos[:, :, idx]           # [B, A, D]
                pred = decoded_preds[:, :, idx, t]     # [B, A, D]
                goal_vec = goal[:, :, idx]             # [B, A, D]
                coeff = coeffs[:, :, idx]              # [B, A, 3]
                
                # Get neighbor positions (all other entities)
                neighbor_idxs = [i for i in range(ent) if i != idx]
                if neighbor_idxs:
                    neighbors_pos = torch.stack(
                        [current_pos[:, :, j] for j in neighbor_idxs],
                        dim=2
                    )  # [B, A, num_neighbors, D]
                else:
                    neighbors_pos = torch.empty(B, A, 0, D, device=device)
                
                # Compute PFM forces
                force, coeff_upd = self.pfm(pos, pred, neighbors_pos, goal_vec, coeff)
                
                # Apply force to get adjusted position
                if t == 0:
                    new_pos = pos + force
                else:
                    new_pos = adjusted_preds[:, :, idx, t - 1] + force
                
                # Apply speed constraints
                if t > 0:
                    prev_pos = adjusted_preds[:, :, idx, t - 1]
                    disp = new_pos - prev_pos
                    speed = torch.norm(disp, dim=-1, keepdim=True)
                    speed_min = v_min_exp.view(B, A, ent)[..., idx].reshape(B * A, 1)
                    speed_max = v_max_exp.view(B, A, ent)[..., idx].reshape(B * A, 1)
                    clipped_speed = torch.clamp(speed, speed_min, speed_max)
                    adj_disp = disp / (speed + 1e-8) * clipped_speed
                    new_pos = prev_pos + adj_disp
                
                adjusted_preds[:, :, idx, t] = new_pos
                
                if idx == 0:
                    coeff_list.append(coeff_upd)
            
            # Update current position for next timestep
            current_pos = adjusted_preds[:, :, :, t].clone()
        
        # Compute coefficient statistics
        coeff_stack = torch.stack(coeff_list)
        coeff_mean = coeff_stack.mean()
        coeff_var = coeff_stack.var(unbiased=False)
        
        return adjusted_preds, decoded_preds, coeff_mean, coeff_var


# =============================================================================
# ALIASES FOR COMPATIBILITY
# =============================================================================
CheckpointedIntegratedMTAPFM_neighbours = LTC_PFM_HybridModel
LiquidPFMHybridModel = LTC_PFM_HybridModel
CfC_PFM_Model = LTC_PFM_HybridModel


# =============================================================================
# ARCHITECTURE DOCUMENTATION
# =============================================================================
"""
================================================================================
                        LTC-PFM HYBRID ARCHITECTURE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                     │
│  history_neighbors: [B, A, ent, H, 2]     goal: [B, A, ent, 2]              │
│      B=batch, A=agents, ent=entities, H=history, 2=(x,y)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT EMBEDDING                                    │
│  nn.Linear(2 → hidden_size)                                                  │
│  Shape: [B*A*ent, H, 2] → [B*A*ent, H, hidden_size]                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CfC ENCODER (Tweakable)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ODE Dynamics:                                                               │
│      dx/dt = -[1/τ + f(x,I,θ)] x + f(x,I,θ) A                               │
│                                                                              │
│  Mode Options:                                                               │
│    • "default": Gated interpolation  x_new = ff1*(1-g) + ff2*g              │
│    • "pure": Direct solution         x_new = -A*exp(-ts*(|wτ|+|ff1|))*ff1+A │
│    • "no_gate": Additive             x_new = ff1 + g*ff2                    │
│                                                                              │
│  Parameters:                                                                 │
│    • backbone_units: MLP hidden size (default 128)                          │
│    • backbone_layers: Number of MLP layers (default 1)                      │
│    • backbone_activation: lecun_tanh, relu, gelu, etc.                      │
│    • mixed_memory: Augment with LSTM cell (default True)                    │
│                                                                              │
│  Output: h_enc [B*A*ent, hidden_size]                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────┐   ┌─────────────────────────────────────────┐
│   COEFFICIENT PROJECTOR       │   │         CfC DECODER (Tweakable)         │
│   nn.Linear(hidden → 3)       │   ├─────────────────────────────────────────┤
│                               │   │  Same ODE dynamics as encoder           │
│   Output: [B, A, ent, 3]      │   │  Autoregressive: T steps                │
│   [k_goal, k_pred, k_rep]     │   │                                         │
│                               │   │  At each step t:                        │
│                               │   │    input_t ← embed(pos_{t-1})           │
│                               │   │    out_t, hx ← CfC(input_t, hx)         │
│                               │   │    residual_t ← Linear(out_t)           │
└───────────────────────────────┘   │                                         │
            │                       │  If use_angular_velocity:               │
            │                       │    residual = (Δv, Δω)                  │
            │                       │    v_t = v_avg + Δv                     │
            │                       │    θ_t = θ_{t-1} + (ω_avg+Δω)*dt        │
            │                       │    pos_t = pos_{t-1} + v_t*[cosθ,sinθ]*dt│
            │                       │                                         │
            │                       │  Output: decoded_preds [B,A,ent,T,2]    │
            │                       └─────────────────────────────────────────┘
            │                                           │
            │                                           ▼
            │               ┌─────────────────────────────────────────────────┐
            │               │              PFM CORRECTION LOOP                 │
            │               ├─────────────────────────────────────────────────┤
            └───────────────│  For t = 1..T:                                  │
                            │    For each entity idx:                         │
                            │      pos = current_pos[:,:,idx]                 │
                            │      pred = decoded_preds[:,:,idx,t]            │
                            │      neighbors = other entities' positions      │
                            │                                                 │
                            │      ┌───────────────────────────────────────┐  │
                            │      │     POTENTIAL FIELD MODULE             │  │
                            │      ├───────────────────────────────────────┤  │
                            │      │ F_goal = k1 * (goal - pos)            │  │
                            │      │ F_pred = k2 * (pred - pos)            │  │
                            │      │ F_rep = Σ kr*(pos-nbr)/||d||²         │  │
                            │      │ force = F_goal + F_pred + F_rep       │  │
                            │      └───────────────────────────────────────┘  │
                            │                                                 │
                            │      new_pos = prev_pos + force                 │
                            │      Apply speed constraints                    │
                            │      adjusted_preds[:,:,idx,t] = new_pos        │
                            │                                                 │
                            │    current_pos = adjusted_preds[:,:,:,t]        │
                            └─────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                    │
│  adjusted_preds: [B, A, ent, T, 2] - Physics-corrected trajectory           │
│  decoded_preds:  [B, A, ent, T, 2] - Raw neural prediction                  │
│  coeff_mean:     Scalar - Mean of PFM coefficients                          │
│  coeff_var:      Scalar - Variance of PFM coefficients                      │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                           TWEAKABLE PARAMETERS
================================================================================

CfC CELL PARAMETERS (in utils/cfc_cell.py):
├── mode: "default" | "pure" | "no_gate"
├── backbone_activation: "lecun_tanh" | "relu" | "gelu" | "tanh" | "hardtanh"
├── backbone_units: int (default 128)
├── backbone_layers: int (default 1)
├── backbone_dropout: float (default 0.0)
└── mixed_memory: bool (augment with LSTM)

In CfCCell.forward():
├── ff1 = Linear(backbone_out)         ← First transformation
├── ff2 = Linear(backbone_out)         ← Second transformation (if not pure)
├── time_a, time_b = Linear(...)       ← Time-dependent gating
├── w_tau = Parameter                  ← Time constant (pure mode)
└── A = Parameter                      ← Attractor state (pure mode)

PFM PARAMETERS:
├── k_goal_init: Initial goal attraction
├── k_pred_init: Initial prediction attraction
├── k_rep_init: Initial repulsion strength
└── repulsion_radius: Distance threshold

KINEMATIC PARAMETERS:
├── target_avg_speed: Expected speed
├── speed_tolerance: Allowed deviation
├── dt: Integration timestep
└── use_angular_velocity: Enable (Δv, Δω) mode

================================================================================
                             EXAMPLE USAGE
================================================================================

```python
from models.ltc_pfm_hybrid_model import LTC_PFM_HybridModel

# Create model with custom CfC parameters
model = LTC_PFM_HybridModel(
    hidden_size=64,
    cfc_mode="default",           # Try "pure" for direct ODE solution
    cfc_backbone_units=128,
    cfc_backbone_layers=1,
    cfc_backbone_activation="lecun_tanh",
    mixed_memory=True,
    use_angular_velocity=True,
    target_avg_speed=4.087,
    dt=0.1
)

# Forward pass
history = torch.randn(B, A, ent, H, 2)
goal = torch.randn(B, A, 2)
adjusted, decoded, coeff_mean, coeff_var = model(history, goal)

# Loss computation
loss = F.mse_loss(adjusted[:, :, 0], ground_truth)  # Ego predictions
loss.backward()
```

================================================================================
                         MODIFYING THE ODE DYNAMICS
================================================================================

To modify the core CfC dynamics, edit utils/cfc_cell.py:

1. Change the time-dependent gate:
   ```python
   # In CfCCell.forward()
   t_interp = self.sigmoid(t_a * ts + t_b)  # Original
   t_interp = self.sigmoid(t_a * ts**2 + t_b)  # Quadratic time dependence
   ```

2. Change the state update:
   ```python
   # Original default mode
   new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
   
   # Custom: Add residual connection
   new_hidden = hx + (ff1 * (1.0 - t_interp) + t_interp * ff2)
   ```

3. Add learnable decay:
   ```python
   # Add to __init__
   self.decay = nn.Parameter(torch.ones(1, hidden_size) * 0.9)
   
   # Use in forward
   new_hidden = self.decay * hx + (1 - self.decay) * computed_hidden
   ```

================================================================================
"""
