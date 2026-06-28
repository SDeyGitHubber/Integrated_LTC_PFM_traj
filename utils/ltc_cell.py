"""
================================================================================
TWEAKABLE LIQUID TIME-CONSTANT (LTC) CELL
================================================================================

This module provides a fully customizable LTC cell implementation based on:
- Hasani et al., "Liquid Time-Constant Networks", AAAI 2021
- Hasani et al., "Neural Circuit Policies", NeurIPS 2020

The LTC cell uses explicit ODE numerical integration (unlike CfC which uses
closed-form approximation).

--------------------------------------------------------------------------------
MATHEMATICAL FOUNDATION (from LTC Paper)
--------------------------------------------------------------------------------

The LTC architecture is based on the continuous differential equation (Eq. 1):

    dx(t)/dt = -[1/τ + f(x(t), I(t), t, θ)] x(t) + f(x(t), I(t), t, θ) A

Expanded form (Eq. 2):

    τᵢ · (dxᵢ/dt) = -xᵢ + Σⱼ wᵢⱼ · σ(xⱼ) + Σₖ wᵢₖ · Iₖ(t)

Where:
    - xᵢ(t): Hidden state of neuron i at time t
    - τᵢ: Time constant of neuron i (learnable)
    - wᵢⱼ: Recurrent weight from neuron j to i
    - wᵢₖ: Input weight from input k to neuron i
    - σ: Activation function (sigmoid/tanh)
    - Iₖ(t): Input signal k at time t

--------------------------------------------------------------------------------
NUMERICAL INTEGRATION (ODE Unfolds)
--------------------------------------------------------------------------------

Unlike CfC's closed-form solution, LTC uses numerical ODE integration:

    For each unfold step n = 1..N:
        1. Compute input contribution:
           sensory_input = σ(W_sensory · I(t) + b_sensory)
        
        2. Compute recurrent contribution:
           recurrent_input = σ(W_recurrent · x + b_recurrent)
        
        3. Compute activation:
           cm_t = σ(W_cm · x + input_term + b_cm)
        
        4. Apply time constant:
           dx = (ts / τ) · (-x + cm_t)
        
        5. Update state:
           x ← x + dx / N

This multi-step integration provides better accuracy but higher compute cost.

--------------------------------------------------------------------------------
KEY THEOREMS (Stability Guarantees)
--------------------------------------------------------------------------------

THEOREM 1 - Bounded Time-Constant:
    τᵢ/(1 + τᵢWᵢ) ≤ τₛᵧₛ,ᵢ ≤ τᵢ
    → System time constant is bounded between explicit τ and effective τ

THEOREM 2 - Bounded Hidden State:
    |x(t)| is bounded for all t ∈ [0, ∞)
    → Network outputs never explode, even with unbounded inputs

COROLLARY - Gradient Bounds:
    Gradients are bounded throughout training
    → No vanishing/exploding gradients

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------

```python
from utils.ltc_cell import LTCCell

cell = LTCCell(
    wiring=FullyConnected(units=64),
    in_features=32,
    input_mapping="affine",
    output_mapping="affine",
    ode_unfolds=6
)

# Single step forward
x_t = torch.randn(batch_size, 32)
h_t = torch.randn(batch_size, 64)
ts = 1.0  # time span

h_out, h_new = cell(x_t, h_t, ts)
```

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


class FullyConnectedWiring:
    """
    Simple fully-connected wiring configuration.
    
    Creates a wiring where all neurons connect to all others.
    """
    def __init__(self, units: int, output_dim: Optional[int] = None):
        self.units = units
        self.input_dim = units
        self.output_dim = output_dim if output_dim else units
        # Full connectivity matrix
        self.adjacency_matrix = np.ones((units, units))
        self.sensory_adjacency_matrix = np.ones((units, units))


class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell with full configurability.
    
    This is an RNNCell that processes single time-steps using ODE
    numerical integration. Unlike CfC, LTC provides explicit control
    over the integration process.
    
    Args:
        wiring: Wiring configuration (FullyConnected or custom)
        in_features (int): Number of input features
        input_mapping (str): "affine", "linear", or None
        output_mapping (str): "affine", "linear", or None  
        ode_unfolds (int): Number of ODE integration steps (default 6)
        epsilon (float): Small constant for numerical stability
        implicit_param_constraints (bool): If True, apply softplus to τ
        
    Mathematical Details:
    --------------------
    
    State update (per ODE unfold):
    
    1. Sensory activation:
       s = σ(W_sensory · I + b_sensory)  if input_mapping="affine"
       s = σ(W_sensory · I)              if input_mapping="linear"
       s = I                              if input_mapping=None
    
    2. Recurrent activation:
       r = σ(W_recurrent · x + b_recurrent)
    
    3. Combined modulation:
       cm = σ(W_cm · s + W_cm_rec · r + b_cm)
    
    4. Time constant application:
       dx = (ts / τ) · (-x + cm · A)
    
    5. State update:
       x ← x + dx / ode_unfolds
    """
    
    def __init__(
        self,
        wiring: Union[FullyConnectedWiring, object],
        in_features: int,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.wiring = wiring
        self.in_features = in_features
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon
        self.implicit_param_constraints = implicit_param_constraints
        
        # Derived dimensions
        self.state_size = wiring.units
        self.motor_size = wiring.output_dim
        
        # ====================================================================
        # SENSORY LAYER (Input Processing)
        # Maps external inputs to sensory activations
        # ====================================================================
        if input_mapping in ["affine", "linear"]:
            self.sensory_weight = nn.Parameter(
                torch.randn(in_features, self.state_size) * 0.1
            )
            if input_mapping == "affine":
                self.sensory_bias = nn.Parameter(
                    torch.zeros(self.state_size)
                )
            else:
                self.register_buffer('sensory_bias', None)
        else:
            self.sensory_weight = None
            self.sensory_bias = None
        
        # Sensory-to-hidden connectivity (erev: reversal potential analog)
        self.sensory_mu = nn.Parameter(
            torch.randn(in_features, self.state_size) * 0.5
        )
        self.sensory_sigma = nn.Parameter(
            torch.ones(in_features, self.state_size) * 0.5
        )
        
        # ====================================================================
        # RECURRENT LAYER (State-to-State Processing)  
        # Models inter-neuron dynamics
        # ====================================================================
        self.recurrent_weight = nn.Parameter(
            torch.randn(self.state_size, self.state_size) * 0.1
        )
        self.recurrent_bias = nn.Parameter(
            torch.zeros(self.state_size)
        )
        
        # Recurrent connectivity parameters
        self.recurrent_mu = nn.Parameter(
            torch.randn(self.state_size, self.state_size) * 0.5
        )
        self.recurrent_sigma = nn.Parameter(
            torch.ones(self.state_size, self.state_size) * 0.5
        )
        
        # ====================================================================
        # TIME CONSTANTS (τ)
        # Controls the speed of neural dynamics
        # ====================================================================
        # Initialize with log(τ) for positivity (τ = exp(log_tau) or softplus)
        self.log_tau = nn.Parameter(
            torch.zeros(self.state_size)  # τ ≈ 1 initially
        )
        
        # ====================================================================
        # ATTRACTOR STATE (A)
        # Equilibrium points the system tends toward
        # ====================================================================
        self.gleak = nn.Parameter(
            torch.ones(self.state_size)  # Leak conductance
        )
        self.vleak = nn.Parameter(
            torch.zeros(self.state_size)  # Leak potential (attractor)
        )
        
        # ====================================================================
        # COMBINED MODULATION
        # Final gating of state updates
        # ====================================================================
        self.cm_weight = nn.Parameter(
            torch.randn(self.state_size, self.state_size) * 0.1
        )
        self.cm_bias = nn.Parameter(
            torch.zeros(self.state_size)
        )
        
        # ====================================================================
        # OUTPUT MAPPING
        # Maps hidden state to output
        # ====================================================================
        if output_mapping in ["affine", "linear"]:
            self.output_weight = nn.Parameter(
                torch.randn(self.state_size, self.motor_size) * 0.1
            )
            if output_mapping == "affine":
                self.output_bias = nn.Parameter(
                    torch.zeros(self.motor_size)
                )
            else:
                self.register_buffer('output_bias', None)
        else:
            self.output_weight = None
            self.output_bias = None
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def get_tau(self) -> torch.Tensor:
        """
        Get time constants with positivity constraint.
        
        Returns τ = softplus(log_tau) + ε  or  exp(log_tau) + ε
        """
        if self.implicit_param_constraints:
            # Softplus: smooth approximation to ReLU
            return torch.nn.functional.softplus(self.log_tau) + self.epsilon
        else:
            # Exponential: strictly positive
            return torch.exp(self.log_tau) + self.epsilon
    
    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        ts: float = 1.0
    ) -> tuple:
        """
        Single timestep forward pass with ODE integration.
        
        Implements the LTC ODE:
            dx/dt = (1/τ) · [-x + f(x, I)]
        
        Using Euler integration with multiple unfold steps.
        
        Args:
            inputs: Input tensor [batch, in_features]
                    I(t) in the LTC equation
            state: Hidden state [batch, state_size]
                   x(t) in the LTC equation
            ts: Time span (scalar or [batch] tensor)
                Δt in the integration
        
        Returns:
            output: Output tensor [batch, motor_size]
            new_state: New hidden state [batch, state_size]
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Get time constants
        tau = self.get_tau()  # [state_size]
        
        # Handle scalar or tensor ts
        if isinstance(ts, (int, float)):
            ts = torch.tensor(ts, device=device, dtype=inputs.dtype)
        if ts.dim() == 0:
            ts = ts.expand(batch_size)
        ts = ts.unsqueeze(-1)  # [batch, 1]
        
        # ====================================================================
        # SENSORY PROCESSING
        # Compute input contribution to dynamics
        # ====================================================================
        if self.sensory_weight is not None:
            # Sensory activation: s = σ(W_s · I + b_s)
            sensory_act = inputs @ self.sensory_weight
            if self.sensory_bias is not None:
                sensory_act = sensory_act + self.sensory_bias
            sensory_act = self.sigmoid(sensory_act)
        else:
            sensory_act = inputs
        
        # Sensory reversal potential contribution
        # Models how inputs influence the attractor
        sensory_input = sensory_act @ self.sensory_mu
        sensory_numerator = sensory_act @ (self.sensory_mu * self.sensory_sigma)
        
        # ====================================================================
        # ODE INTEGRATION (Multiple Unfolds)
        # ====================================================================
        current_state = state.clone()
        
        for unfold in range(self.ode_unfolds):
            # ----------------------------------------------------------------
            # Recurrent Processing
            # Compute state-to-state contribution
            # ----------------------------------------------------------------
            recurrent_act = current_state @ self.recurrent_weight + self.recurrent_bias
            recurrent_act = self.sigmoid(recurrent_act)
            
            # Recurrent reversal potential contribution
            recurrent_input = recurrent_act @ self.recurrent_mu
            recurrent_numerator = recurrent_act @ (self.recurrent_mu * self.recurrent_sigma)
            
            # ----------------------------------------------------------------
            # Combined Modulation
            # Gate the state update
            # ----------------------------------------------------------------
            cm_input = current_state @ self.cm_weight + self.cm_bias
            cm_gate = self.sigmoid(cm_input)
            
            # ----------------------------------------------------------------
            # Compute Total Input Current
            # ----------------------------------------------------------------
            # Total conductance
            total_w = (
                torch.abs(self.gleak).unsqueeze(0) +
                sensory_input +
                recurrent_input
            )
            
            # Total weighted input (numerator of activation)
            total_numerator = (
                self.vleak.unsqueeze(0) * torch.abs(self.gleak).unsqueeze(0) +
                sensory_numerator +
                recurrent_numerator
            )
            
            # Effective input (attractor target)
            effective_input = total_numerator / (total_w + self.epsilon)
            effective_input = effective_input * cm_gate
            
            # ----------------------------------------------------------------
            # State Update (Euler Integration)
            # dx/dt = (1/τ) · (-x + effective_input)
            # x_new = x + (ts/ode_unfolds) · dx/dt
            # ----------------------------------------------------------------
            dx_dt = (-current_state + effective_input) / tau.unsqueeze(0)
            delta_t = ts / self.ode_unfolds
            current_state = current_state + delta_t * dx_dt
        
        new_state = current_state
        
        # ====================================================================
        # OUTPUT MAPPING
        # ====================================================================
        if self.output_weight is not None:
            output = new_state @ self.output_weight
            if self.output_bias is not None:
                output = output + self.output_bias
        else:
            output = new_state
        
        return output, new_state
    
    def get_ode_components(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor
    ) -> dict:
        """
        Extract interpretable ODE components for analysis.
        
        Returns:
            dict with:
                - 'tau': Time constants
                - 'sensory_act': Sensory layer activation
                - 'recurrent_act': Recurrent layer activation
                - 'cm_gate': Combined modulation gate
                - 'effective_input': Computed attractor target
                - 'gleak': Leak conductance
                - 'vleak': Leak potential (attractor)
        """
        tau = self.get_tau()
        
        # Sensory
        if self.sensory_weight is not None:
            sensory_act = inputs @ self.sensory_weight
            if self.sensory_bias is not None:
                sensory_act = sensory_act + self.sensory_bias
            sensory_act = self.sigmoid(sensory_act)
        else:
            sensory_act = inputs
        
        sensory_input = sensory_act @ self.sensory_mu
        sensory_numerator = sensory_act @ (self.sensory_mu * self.sensory_sigma)
        
        # Recurrent
        recurrent_act = state @ self.recurrent_weight + self.recurrent_bias
        recurrent_act = self.sigmoid(recurrent_act)
        
        recurrent_input = recurrent_act @ self.recurrent_mu
        recurrent_numerator = recurrent_act @ (self.recurrent_mu * self.recurrent_sigma)
        
        # CM gate
        cm_input = state @ self.cm_weight + self.cm_bias
        cm_gate = self.sigmoid(cm_input)
        
        # Total
        total_w = torch.abs(self.gleak).unsqueeze(0) + sensory_input + recurrent_input
        total_numerator = (
            self.vleak.unsqueeze(0) * torch.abs(self.gleak).unsqueeze(0) +
            sensory_numerator + recurrent_numerator
        )
        effective_input = total_numerator / (total_w + self.epsilon) * cm_gate
        
        return {
            'tau': tau,
            'sensory_act': sensory_act,
            'recurrent_act': recurrent_act,
            'cm_gate': cm_gate,
            'effective_input': effective_input,
            'gleak': self.gleak,
            'vleak': self.vleak
        }


class LTC(nn.Module):
    """
    Liquid Time-Constant RNN wrapper.
    
    Applies an LTC cell to process sequences, supporting:
    - Variable length sequences via timespans
    - Mixed memory mode (augmented with LSTM cell)
    - Sequence or final-output return modes
    
    Args:
        input_size: Number of input features
        units: Number of hidden units
        return_sequences: If True, return outputs at all timesteps
        batch_first: If True, input shape is (batch, seq, features)
        mixed_memory: If True, augment with LSTM for long-term memory
        input_mapping: Input layer type ("affine", "linear", None)
        output_mapping: Output layer type ("affine", "linear", None)
        ode_unfolds: Number of integration steps per timestep
        epsilon: Numerical stability constant
    """
    
    def __init__(
        self,
        input_size: int,
        units: int,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.use_mixed = mixed_memory
        
        # Create wiring
        wiring = FullyConnectedWiring(units)
        
        # Core LTC cell
        self.rnn_cell = LTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon
        )
        
        # Optional LSTM for mixed memory
        if self.use_mixed:
            self.lstm_cell = nn.LSTMCell(input_size, units)
    
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
        Process input sequence through LTC.
        
        Args:
            input: Input tensor (see CfC for shape details)
            hx: Initial hidden state
            timespans: Time spans per step
        
        Returns:
            output: Sequence or final output
            hx: Final hidden state
        """
        device = input.device
        
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
                    raise RuntimeError("LTC with mixed_memory requires (h0, c0) tuple")
                h_state, c_state = hx
            else:
                h_state = hx
                c_state = None
            
            if not is_batched:
                h_state = h_state.unsqueeze(0)
                if c_state is not None:
                    c_state = c_state.unsqueeze(0)
        
        # Process sequence
        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()
            
            if self.use_mixed:
                h_state, c_state = self.lstm_cell(inputs, (h_state, c_state))
            
            h_out, h_state = self.rnn_cell(inputs, h_state, ts)
            
            if self.return_sequences:
                output_sequence.append(h_out)
        
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        
        hx = (h_state, c_state) if self.use_mixed else h_state
        
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]
        
        return readout, hx
