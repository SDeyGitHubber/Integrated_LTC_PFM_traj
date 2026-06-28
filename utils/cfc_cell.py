"""
================================================================================
TWEAKABLE CLOSED-FORM CONTINUOUS-TIME (CfC) CELL
================================================================================

This module provides a fully customizable CfC cell implementation based on:
- Hasani et al., "Closed-form Continuous-time Neural Networks", Nature ML 2022
- Hasani et al., "Liquid Time-Constant Networks", AAAI 2021

The CfC cell is the modern, solver-free evolution of the LTC architecture.

--------------------------------------------------------------------------------
MATHEMATICAL FOUNDATION (from LTC Paper, Eq. 1-3)
--------------------------------------------------------------------------------

The LTC/CfC architecture is based on the continuous differential equation:

    dx(t)/dt = -[1/τ + f(x(t), I(t), t, θ)] x(t) + f(x(t), I(t), t, θ) A    (Eq. 1)

Where:
    - x(t): hidden state (context vector) at time t
    - τ: time constant (controls decay rate)
    - f(·): bounded, sigmoidal nonlinearity ensuring stability
    - I(t): input features (position, velocity, etc.)
    - A: learnable attractor state (equilibrium point)
    - θ: learnable parameters

The update step (Eq. 3) is computed as:

    x(t + Δt) = [x(t) + Δt·f(·)·A] / [1 + Δt·(1/τ + f(·))]               (Eq. 3)

--------------------------------------------------------------------------------
KEY THEOREMS (Stability Guarantees)
--------------------------------------------------------------------------------

THEOREM 1 - Bounded Time-Constant:
    τᵢ/(1 + τᵢWᵢ) ≤ τₛᵧₛ,ᵢ ≤ τᵢ
    → Processing speed is controlled and stable

THEOREM 2 - Bounded Hidden State:
    |x(t)| is bounded for all t
    → Network outputs never explode, even with unbounded inputs

--------------------------------------------------------------------------------
CfC MODES
--------------------------------------------------------------------------------

1. DEFAULT MODE (with gating):
   - Uses interpolation gating: g = σ(t_a * ts + t_b)
   - new_hidden = ff1 * (1 - g) + ff2 * g
   - Best balance of expressivity and stability

2. PURE MODE (direct solution):
   - Direct solution approximation without gating
   - new_hidden = -A * exp(-ts * (|w_τ| + |ff1|)) * ff1 + A
   - Closest to original ODE solution

3. NO_GATE MODE:
   - Additive combination without gating
   - new_hidden = ff1 + g * ff2
   - Simpler but less expressive

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------

```python
from utils.cfc_cell import CfCCell

cell = CfCCell(
    input_size=64,
    hidden_size=64,
    mode="default",
    backbone_activation="lecun_tanh",
    backbone_units=128,
    backbone_layers=1
)

# Single step forward
x_t = torch.randn(batch_size, input_size)
h_t = torch.randn(batch_size, hidden_size)
ts = 1.0  # time span

h_out, h_new = cell(x_t, h_t, ts)
```

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class LeCunTanh(nn.Module):
    """
    LeCun's scaled tanh activation function.
    
    f(x) = 1.7159 * tanh(0.666 * x)
    
    Properties:
    - Zero-centered
    - Near-unit variance for normalized inputs
    - Smoother gradients than standard tanh
    - Recommended for CfC backbone layers
    """
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) Cell with full configurability.
    
    This is an RNNCell that processes single time-steps. The cell implements
    the closed-form solution to the LTC differential equation.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units (context dimension)
        mode (str): Operation mode - "default", "pure", or "no_gate"
        backbone_activation (str): Activation for backbone layers
        backbone_units (int): Hidden units in backbone MLP
        backbone_layers (int): Number of backbone layers
        backbone_dropout (float): Dropout rate in backbone
        sparsity_mask (np.ndarray): Optional sparsity mask for connections
        
    Mathematical Details:
    --------------------
    
    For mode="default" (gated interpolation):
        1. Concatenate input and hidden: z = [I(t), x(t)]
        2. Process through backbone: f = backbone(z)
        3. Compute two transformation branches:
           ff1 = tanh(W₁ · f + b₁)
           ff2 = tanh(W₂ · f + b₂)
        4. Compute time-dependent gate:
           g = σ(t_a · f · ts + t_b · f)
        5. Interpolate:
           x(t+Δt) = ff1 · (1 - g) + ff2 · g
    
    For mode="pure" (direct ODE solution):
        x(t+Δt) = -A · exp(-ts · (|w_τ| + |ff1|)) · ff1 + A
        
        Where:
        - A: learnable attractor state
        - w_τ: learnable time constant weights
        - ts: time span
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "default",
        backbone_activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        sparsity_mask: Optional[np.ndarray] = None
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Validate mode
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(f"Unknown mode '{mode}', valid options: {allowed_modes}")
        self.mode = mode
        
        # Setup sparsity mask (for NCP-style sparse connectivity)
        if sparsity_mask is not None:
            self.sparsity_mask = nn.Parameter(
                torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)),
                requires_grad=False
            )
        else:
            self.sparsity_mask = None
        
        # Select activation function
        activation_map = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "lecun_tanh": LeCunTanh,
            "hardtanh": nn.Hardtanh,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU
        }
        if backbone_activation not in activation_map:
            raise ValueError(f"Unknown activation '{backbone_activation}'")
        act_fn = activation_map[backbone_activation]
        
        # ====================================================================
        # BACKBONE NETWORK
        # Processes concatenated [input, hidden] through MLP
        # f(x(t), I(t)) in the LTC equation
        # ====================================================================
        self.backbone = None
        self.backbone_layers = backbone_layers
        
        if backbone_layers > 0:
            layers = [
                nn.Linear(input_size + hidden_size, backbone_units),
                act_fn()
            ]
            for i in range(1, backbone_layers):
                layers.append(nn.Linear(backbone_units, backbone_units))
                layers.append(act_fn())
                if backbone_dropout > 0.0:
                    layers.append(nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layers)
        
        # Activation functions for output processing
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # Input dimension for output layers
        cat_shape = hidden_size + input_size if backbone_layers == 0 else backbone_units
        
        # ====================================================================
        # OUTPUT TRANSFORMATION LAYERS
        # Maps backbone features to hidden state updates
        # ====================================================================
        
        # First transformation: ff1 = tanh(W₁·f + b₁)
        self.ff1 = nn.Linear(cat_shape, hidden_size)
        
        if self.mode == "pure":
            # Pure mode: Direct ODE solution approximation
            # x(t+Δt) = -A · exp(-ts · (|w_τ| + |ff1|)) · ff1 + A
            
            # w_τ: Time constant weights (learnable)
            # Controls the decay rate of the ODE
            self.w_tau = nn.Parameter(
                torch.zeros(1, hidden_size),
                requires_grad=True
            )
            
            # A: Attractor state (learnable)
            # The equilibrium point the system tends toward
            self.A = nn.Parameter(
                torch.ones(1, hidden_size),
                requires_grad=True
            )
        else:
            # Default/no_gate mode: Gated interpolation
            
            # Second transformation: ff2 = tanh(W₂·f + b₂)
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            
            # Time-dependent gating mechanism
            # g = σ(t_a·f·ts + t_b·f)
            self.time_a = nn.Linear(cat_shape, hidden_size)  # Time scaling
            self.time_b = nn.Linear(cat_shape, hidden_size)  # Time bias
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for all weight matrices."""
        for param in self.parameters():
            if param.dim() == 2 and param.requires_grad:
                nn.init.xavier_uniform_(param)
    
    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor,
        ts: float = 1.0
    ) -> tuple:
        """
        Single timestep forward pass.
        
        Implements the closed-form solution to the LTC ODE:
        
        dx/dt = -[1/τ + f(·)]x + f(·)A
        
        Args:
            input: Input tensor [batch, input_size]
                   I(t) in the LTC equation
            hx: Hidden state [batch, hidden_size]
                x(t) in the LTC equation
            ts: Time span (scalar or [batch] tensor)
                Δt in the update equation
        
        Returns:
            h_out: Output tensor [batch, hidden_size]
            h_new: New hidden state [batch, hidden_size]
        
        Mathematical Flow:
        -----------------
        1. z = concat([I(t), x(t)])           # Combine input and state
        2. f = backbone(z)                     # Extract features
        3. Apply mode-specific update          # Compute x(t+Δt)
        """
        # Step 1: Concatenate input and hidden state
        # z = [I(t), x(t)] ∈ ℝ^(input_size + hidden_size)
        x = torch.cat([input, hx], dim=1)
        
        # Step 2: Process through backbone (if present)
        # f(x(t), I(t)) ∈ ℝ^backbone_units
        if self.backbone_layers > 0:
            x = self.backbone(x)
        
        # Step 3: Compute first transformation
        # ff1 = W₁·f + b₁ (before activation)
        if self.sparsity_mask is not None:
            ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(x)
        
        # Step 4: Mode-specific update
        if self.mode == "pure":
            # ================================================================
            # PURE MODE: Direct ODE Solution
            # ================================================================
            # 
            # From the LTC ODE:
            #   dx/dt = -[1/τ + f(·)]x + f(·)A
            # 
            # The closed-form solution approximation:
            #   x(t+Δt) ≈ -A · exp(-ts · (|w_τ| + |ff1|)) · ff1 + A
            # 
            # Interpretation:
            # - exp(-ts · (|w_τ| + |ff1|)): Decay factor controlled by
            #   time constant w_τ and input-dependent term ff1
            # - Multiplied by ff1: Input-modulated decay
            # - +A: Attractor state (equilibrium)
            # ================================================================
            
            new_hidden = (
                -self.A 
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
            
        else:
            # ================================================================
            # DEFAULT/NO_GATE MODE: Gated Interpolation
            # ================================================================
            # 
            # Two branches:
            #   ff1 = tanh(W₁·f + b₁)  - Primary transformation
            #   ff2 = tanh(W₂·f + b₂)  - Secondary transformation
            # 
            # Time-dependent gate:
            #   g = σ(t_a·ts + t_b)    - Sigmoid gating
            # 
            # Interpolation:
            #   default:  x(t+Δt) = ff1·(1-g) + ff2·g  (blending)
            #   no_gate:  x(t+Δt) = ff1 + g·ff2        (additive)
            # ================================================================
            
            # Second transformation branch
            if self.sparsity_mask is not None:
                ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(x)
            
            # Apply activations
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            
            # Time-dependent gating
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            
            if self.mode == "no_gate":
                # Additive combination
                new_hidden = ff1 + t_interp * ff2
            else:  # default
                # Gated interpolation (blending)
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        
        # Output is same as new hidden state (can add projection if needed)
        return new_hidden, new_hidden
    
    def get_ode_components(
        self,
        input: torch.Tensor,
        hx: torch.Tensor
    ) -> dict:
        """
        Extract interpretable ODE components for analysis/debugging.
        
        Returns:
            dict with:
                - 'backbone_features': Output of backbone network
                - 'ff1': First transformation output
                - 'ff2': Second transformation (if applicable)
                - 'time_gate': Time-dependent gate values (if applicable)
                - 'w_tau': Time constant weights (if pure mode)
                - 'A': Attractor state (if pure mode)
        """
        x = torch.cat([input, hx], dim=1)
        if self.backbone_layers > 0:
            x = self.backbone(x)
        
        components = {
            'backbone_features': x,
            'ff1': self.ff1(x)
        }
        
        if self.mode == "pure":
            components['w_tau'] = self.w_tau
            components['A'] = self.A
        else:
            components['ff2'] = self.ff2(x)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            components['time_gate'] = self.sigmoid(t_a + t_b)
        
        return components


class CfC(nn.Module):
    """
    Closed-form Continuous-time RNN wrapper.
    
    Applies a CfC cell to process sequences, supporting:
    - Variable length sequences via timespans
    - Mixed memory mode (augmented with LSTM cell)
    - Sequence or final-output return modes
    
    This wrapper provides the same API as ncps.torch.CfC but with
    full access to the internal cell implementation.
    
    Args:
        input_size: Number of input features
        units: Number of hidden units
        proj_size: Optional output projection size
        return_sequences: If True, return outputs at all timesteps
        batch_first: If True, input shape is (batch, seq, features)
        mixed_memory: If True, augment with LSTM for long-term memory
        mode: CfC cell mode ("default", "pure", "no_gate")
        activation: Backbone activation function
        backbone_units: Hidden units in backbone
        backbone_layers: Number of backbone layers
        backbone_dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_size: int,
        units: int,
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.use_mixed = mixed_memory
        
        # Core CfC cell
        self.rnn_cell = CfCCell(
            input_size=input_size,
            hidden_size=units,
            mode=mode,
            backbone_activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout
        )
        
        # Optional LSTM for mixed memory mode
        if self.use_mixed:
            self.lstm_cell = nn.LSTMCell(input_size, units)
        
        # Optional output projection
        if proj_size is not None:
            self.fc = nn.Linear(units, proj_size)
        else:
            self.fc = nn.Identity()
    
    @property
    def state_size(self):
        return self.hidden_size
    
    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Process input sequence through CfC.
        
        Args:
            input: Input tensor
                - (L, C) unbatched
                - (B, L, C) if batch_first=True
                - (L, B, C) if batch_first=False
            hx: Initial hidden state
                - If mixed_memory=False: (B, H) tensor
                - If mixed_memory=True: tuple ((B, H), (B, H))
            timespans: Time spans per step (optional)
                - Same shape as input without feature dim
        
        Returns:
            output: Sequence output or final output
            hx: Final hidden state
        """
        device = input.device
        
        # Handle unbatched input
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)
        
        batch_size = input.size(batch_dim)
        seq_len = input.size(seq_dim)
        
        # Initialize hidden state
        if hx is None:
            h_state = torch.zeros(batch_size, self.hidden_size, device=device)
            c_state = (
                torch.zeros(batch_size, self.hidden_size, device=device)
                if self.use_mixed else None
            )
        else:
            if self.use_mixed:
                if isinstance(hx, torch.Tensor):
                    raise RuntimeError(
                        "CfC with mixed_memory=True requires (h0, c0) tuple"
                    )
                h_state, c_state = hx
            else:
                h_state = hx
                c_state = None
            
            if is_batched and h_state.dim() != 2:
                raise RuntimeError(f"Expected 2D hx for batched input, got {h_state.dim()}D")
            if not is_batched:
                h_state = h_state.unsqueeze(0)
                if c_state is not None:
                    c_state = c_state.unsqueeze(0)
        
        # Process sequence
        output_sequence = []
        for t in range(seq_len):
            # Extract current timestep
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()
            
            # Optional LSTM update (mixed memory)
            if self.use_mixed:
                h_state, c_state = self.lstm_cell(inputs, (h_state, c_state))
            
            # CfC update
            h_out, h_state = self.rnn_cell(inputs, h_state, ts)
            
            if self.return_sequences:
                output_sequence.append(self.fc(h_out))
        
        # Format output
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_out)
        
        hx = (h_state, c_state) if self.use_mixed else h_state
        
        # Handle unbatched output
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]
        
        return readout, hx
