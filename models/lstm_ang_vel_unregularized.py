"""
Un-Regularized Velocity-Based LSTM

This version removes the excessive regularization from the original VelocityBasedLSTM:
1. NO residual parameterization - predicts FULL velocities
2. NO speed clamping - allows arbitrary velocities
3. NO forced omega_avg=0 - learns full angular dynamics
4. Autoregressive decoder feedback - richer temporal modeling
5. Position loss priority - NOT residual minimization

Use this for learning complex, non-linear trajectory patterns.
"""

import torch
import torch.nn as nn


class VelocityBasedLSTMUnregularized(nn.Module):
    """
    Un-regularized LSTM for trajectory prediction.
    
    Key Differences from Original:
    ------------------------------
    - Predicts ABSOLUTE velocities (v, ω), not residuals (Δv, Δω)
    - NO speed clamping constraints
    - NO omega_avg=0 bias (learns turning dynamics freely)
    - Decoder receives previous predictions (autoregressive)
    - Trained with position MSE, not residual norm
    """
    
    def __init__(
        self, 
        input_size=2, 
        hidden_size=64, 
        num_layers=2, 
        dt=0.25,
        output_mode='velocities',  # 'velocities' or 'positions'
        use_speed_limits=False,     # Whether to apply soft speed limits
        max_speed=10.0              # Only used if use_speed_limits=True
    ):
        """
        Initialize the un-regularized LSTM.
        
        Args:
            input_size (int): Input dimension (2 for x, y)
            hidden_size (int): LSTM hidden size
            num_layers (int): Number of LSTM layers
            dt (float): Time step for integration (seconds)
            output_mode (str): 'velocities' (predict v, ω) or 'positions' (direct)
            use_speed_limits (bool): Whether to apply soft speed constraints
            max_speed (float): Maximum speed if limits are used
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dt = dt
        self.output_mode = output_mode
        self.use_speed_limits = use_speed_limits
        self.max_speed = max_speed
        
        # Input embedding
        self.input_embed = nn.Linear(input_size, hidden_size)
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Decoder LSTM (autoregressive)
        self.decoder = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Output heads
        if output_mode == 'velocities':
            # Predict (v, ω) - velocity magnitude and angular velocity
            self.output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)  # (v, ω)
            )
        else:
            # Direct position displacement prediction
            self.output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)  # (Δx, Δy)
            )
    
    def forward(self, history_neighbors, expanded_goals=None, neighbors=None):
        """
        Forward pass - UN-REGULARIZED prediction.
        
        Args:
            history_neighbors: Tensor (B, A, N, H, 2)
            expanded_goals: Optional
            neighbors: Optional
            
        Returns:
            positions: Tensor (B, A, T, 2) - predicted positions
            velocities: Tensor (B, A, T, 2) - predicted (v, ω) if mode='velocities'
            None: Placeholder
        """
        # Extract ego history
        history = history_neighbors[:, :, 0, :, :]  # (B, A, H, 2)
        B, A, H, D = history.shape
        device = history.device
        
        # Flatten for LSTM processing
        hist_flat = history.reshape(B * A, H, D)  # (B*A, H, 2)
        
        # ====================================================================
        # ENCODER: History → Context
        # ====================================================================
        hist_embedded = self.input_embed(hist_flat)  # (B*A, H, hidden_size)
        _, (h_n, c_n) = self.encoder(hist_embedded)
        
        # ====================================================================
        # DECODER: Autoregressive Prediction (NO REGULARIZATION)
        # ====================================================================
        T = 12  # Number of prediction steps
        hx = (h_n, c_n)
        
        if self.output_mode == 'velocities':
            # Predict FULL velocities (v, ω), not residuals
            velocities = []
            
            # Get last position and heading from history
            last_pos = history[:, :, -1, :]  # (B, A, 2)
            last_pos_flat = last_pos.reshape(B * A, 2)
            
            # Compute initial heading from history
            if H > 1:
                displacement = history[:, :, -1, :] - history[:, :, -2, :]
                dx = displacement[:, :, 0]
                dy = displacement[:, :, 1]
                current_theta = torch.atan2(dy, dx)  # (B, A)
            else:
                current_theta = torch.zeros(B, A, device=device)
            
            current_theta_flat = current_theta.reshape(B * A)
            
            # Start decoder with last history embedding (autoregressive seed)
            decoder_input = hist_embedded[:, -1:, :]  # (B*A, 1, hidden_size)
            
            # Storage
            positions = torch.zeros(B * A, T, 2, device=device)
            current_pos = last_pos_flat.clone()
            
            for t in range(T):
                # LSTM step
                out, hx = self.decoder(decoder_input, hx)
                
                # Predict ABSOLUTE velocity (v, ω) - NO residuals
                velocity_output = self.output(out.squeeze(1))  # (B*A, 2)
                
                v_pred = velocity_output[:, 0]      # Speed magnitude
                omega_pred = velocity_output[:, 1]  # Angular velocity
                
                # Optional: Soft speed limits (but NO hard clamping)
                if self.use_speed_limits:
                    v_pred = torch.tanh(v_pred / self.max_speed) * self.max_speed
                else:
                    v_pred = torch.relu(v_pred)  # Just ensure non-negative
                
                # NO omega constraints - full turning freedom
                
                # Integrate velocities to positions
                current_theta_flat = current_theta_flat + omega_pred * self.dt
                current_theta_flat = torch.atan2(
                    torch.sin(current_theta_flat), 
                    torch.cos(current_theta_flat)
                )
                
                dx = v_pred * torch.cos(current_theta_flat) * self.dt
                dy = v_pred * torch.sin(current_theta_flat) * self.dt
                current_pos = current_pos + torch.stack([dx, dy], dim=-1)
                
                positions[:, t] = current_pos
                velocities.append(torch.stack([v_pred, omega_pred], dim=-1).unsqueeze(1))
                
                # AUTOREGRESSIVE FEEDBACK: Use decoder output for next step
                decoder_input = out
            
            positions = positions.view(B, A, T, 2)
            velocities = torch.cat(velocities, dim=1).view(B, A, T, 2)
            
            return positions, velocities, None
            
        else:  # Direct position prediction
            # Predict position displacements directly
            positions = []
            last_pos = history[:, :, -1, :]  # (B, A, 2)
            last_pos_flat = last_pos.reshape(B * A, 2)
            
            decoder_input = hist_embedded[:, -1:, :]
            current_pos = last_pos_flat.clone()
            
            for t in range(T):
                out, hx = self.decoder(decoder_input, hx)
                
                # Predict displacement (Δx, Δy) directly
                displacement = self.output(out.squeeze(1))  # (B*A, 2)
                
                current_pos = current_pos + displacement
                positions.append(current_pos.view(B * A, 1, 2))
                
                # Autoregressive feedback
                decoder_input = out
            
            positions = torch.cat(positions, dim=1)  # (B*A, T, 2)
            positions = positions.view(B, A, T, 2)
            
            return positions, None, None


# Alias for compatibility
DeepLSTMVelocityModelUnregularized = VelocityBasedLSTMUnregularized
