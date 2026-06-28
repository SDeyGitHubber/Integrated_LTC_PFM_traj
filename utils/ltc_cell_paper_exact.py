"""
================================================================================
LTC CELL - EXACT PAPER IMPLEMENTATION (AAAI 2021)
================================================================================

This module implements the EXACT Algorithm 1 from the LTC paper:
"Liquid Time-Constant Networks", Hasani et al., AAAI 2021

Key differences from ltc_cell.py:
1. Uses FUSED ODE SOLVER (Equation 3) instead of explicit Euler
2. Simple attractor A (bias vector) instead of reversal potentials
3. Single f(·) function instead of separate sensory/recurrent/cm layers
4. Matches paper Algorithm 1 line-by-line

================================================================================
PAPER EQUATION 1 (Core LTC Formula)
================================================================================

dx(t)/dt = -[1/τ + f(x(t), I(t), t, θ)] x(t) + f(x(t), I(t), t, θ) A

Where:
- τ = time constant (learnable per neuron)
- f(·) = tanh(W_r·x + W_s·I + b)  [nonlinear function]
- A = attractor state (learnable bias vector)
- x(t) = hidden state
- I(t) = input

================================================================================
PAPER EQUATION 3 (Fused ODE Solver)
================================================================================

x(t + Δt) = [x(t) + Δt·f(·)·A] / [1 + Δt(1/τ + f(·))]

This is the closed-form solution when applying fused implicit-explicit Euler.
All operations are element-wise.

================================================================================
PAPER ALGORITHM 1 (LTC Update)
================================================================================

Parameters: τ, W_r, W_s, b, A, L=unfold_steps
Function: FusedStep(x(t), I(t), Δt, θ)
  return [x(t) + Δt·f(·)·A] / [1 + Δt(1/τ + f(·))]

Main Loop:
for i = 1 to L:
  x(t+Δt) = FusedStep(x(t), I(t), Δt/L, θ)

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


class FullyConnectedWiring:
    """
    Simple fully-connected wiring configuration.
    """
    def __init__(self, units: int, output_dim: Optional[int] = None):
        self.units = units
        self.input_dim = units
        self.output_dim = output_dim if output_dim else units
        self.adjacency_matrix = np.ones((units, units))
        self.sensory_adjacency_matrix = np.ones((units, units))


class LTCCellPaperExact(nn.Module):
    """
    Liquid Time-Constant Cell - EXACT Paper Implementation (AAAI 2021).
    
    Implements Algorithm 1 from the paper with:
    - Fused ODE solver (Equation 3)
    - Simple nonlinearity f = tanh(W_r·x + W_s·I + b)
    - Simple attractor A (bias vector)
    
    Args:
        wiring: Wiring configuration
        in_features (int): Number of input features
        ode_unfolds (int): Number of discretization steps L (default 6)
        epsilon (float): Numerical stability constant
        implicit_param_constraints (bool): If True, apply softplus to τ
    """
    
    def __init__(
        self,
        wiring: Union[FullyConnectedWiring, object],
        in_features: int,
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = True
    ):
        super().__init__()
        
        self.wiring = wiring
        self.in_features = in_features
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon
        self.implicit_param_constraints = implicit_param_constraints
        
        self.state_size = wiring.units
        self.motor_size = wiring.output_dim
        
        # ====================================================================
        # PAPER PARAMETERS (from Algorithm 1)
        # ====================================================================
        
        # τ: Time constants (must be positive)
        self.log_tau = nn.Parameter(
            torch.zeros(self.state_size)
        )
        
        # W_r: Recurrent weights [state_size × state_size]
        self.W_r = nn.Parameter(
            torch.randn(self.state_size, self.state_size) * 0.1
        )
        
        # W_s: Sensory (input) weights [in_features × state_size]
        self.W_s = nn.Parameter(
            torch.randn(in_features, self.state_size) * 0.1
        )
        
        # b: Bias vector [state_size]
        self.b = nn.Parameter(
            torch.zeros(self.state_size)
        )
        
        # A: Attractor state [state_size]
        self.A = nn.Parameter(
            torch.zeros(self.state_size)
        )
        
        # Optional output projection
        self.output_weight = nn.Parameter(
            torch.randn(self.state_size, self.motor_size) * 0.1
        )
        self.output_bias = nn.Parameter(
            torch.zeros(self.motor_size)
        )
        
        # Activation function
        self.tanh = nn.Tanh()
    
    def get_tau(self) -> torch.Tensor:
        """
        Get time constants τ with positivity constraint.
        """
        if self.implicit_param_constraints:
            return torch.nn.functional.softplus(self.log_tau) + self.epsilon
        else:
            return torch.exp(self.log_tau) + self.epsilon
    
    def compute_f(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute nonlinearity f(x(t), I(t), t, θ).
        
        Paper definition:
        f = tanh(W_r·x + W_s·I + b)
        
        Args:
            inputs: [batch, in_features]
            state: [batch, state_size]
        
        Returns:
            f: [batch, state_size]
        """
        # Recurrent contribution
        recurrent_term = state @ self.W_r
        
        # Sensory contribution
        sensory_term = inputs @ self.W_s
        
        # Combined nonlinearity
        f = self.tanh(recurrent_term + sensory_term + self.b)
        
        return f
    
    def fused_step(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Single fused ODE solver step (Paper Equation 3).
        
        x(t + Δt) = [x(t) + Δt·f(·)·A] / [1 + Δt(1/τ + f(·))]
        
        Args:
            inputs: [batch, in_features]
            state: [batch, state_size]
            dt: [batch, 1] time step
        
        Returns:
            new_state: [batch, state_size]
        """
        # Get time constants
        tau = self.get_tau()  # [state_size]
        
        # Compute nonlinearity f(·)
        f = self.compute_f(inputs, state)  # [batch, state_size]
        
        # ================================================================
        # PAPER EQUATION 3 (Fused ODE Solver)
        # ================================================================
        # Numerator: x(t) + Δt·f(·)·A
        numerator = state + dt * f * self.A.unsqueeze(0)
        
        # Denominator: 1 + Δt(1/τ + f(·))
        denominator = 1.0 + dt * ((1.0 / tau.unsqueeze(0)) + f)
        
        # New state (element-wise division)
        new_state = numerator / (denominator + self.epsilon)
        
        return new_state
    
    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        ts: float = 1.0
    ) -> tuple:
        """
        Forward pass implementing Paper Algorithm 1.
        
        Args:
            inputs: [batch, in_features]
            state: [batch, state_size]
            ts: Time span Δt
        
        Returns:
            output: [batch, motor_size]
            new_state: [batch, state_size]
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Handle time span
        if isinstance(ts, (int, float)):
            ts = torch.tensor(ts, device=device, dtype=inputs.dtype)
        if ts.dim() == 0:
            ts = ts.expand(batch_size)
        ts = ts.unsqueeze(-1)  # [batch, 1]
        
        # ================================================================
        # PAPER ALGORITHM 1
        # ================================================================
        # for i = 1 to L:
        #   x(t+Δt) = FusedStep(x(t), I(t), Δt/L, θ)
        # ================================================================
        
        current_state = state.clone()
        dt = ts / self.ode_unfolds  # Δt/L
        
        for unfold in range(self.ode_unfolds):
            current_state = self.fused_step(inputs, current_state, dt)
        
        new_state = current_state
        
        # Output projection
        output = new_state @ self.output_weight + self.output_bias
        
        return output, new_state
    
    def get_ode_components(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor
    ) -> dict:
        """
        Extract paper components for analysis.
        
        Returns:
            dict with:
                - 'tau': Time constants τ
                - 'f': Nonlinearity f(x,I)
                - 'A': Attractor state
                - 'W_r': Recurrent weights
                - 'W_s': Sensory weights
        """
        return {
            'tau': self.get_tau(),
            'f': self.compute_f(inputs, state),
            'A': self.A,
            'W_r': self.W_r,
            'W_s': self.W_s,
            'b': self.b
        }


class LTCPaperExact(nn.Module):
    """
    LTC RNN wrapper using exact paper implementation.
    
    Args:
        input_size: Number of input features
        units: Number of hidden units
        return_sequences: If True, return outputs at all timesteps
        batch_first: If True, input shape is (batch, seq, features)
        ode_unfolds: Number of integration steps L (default 6)
        epsilon: Numerical stability constant
    """
    
    def __init__(
        self,
        input_size: int,
        units: int,
        return_sequences: bool = True,
        batch_first: bool = True,
        ode_unfolds: int = 6,
        epsilon: float = 1e-8
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        
        # Create wiring
        wiring = FullyConnectedWiring(units)
        
        # Core LTC cell (paper exact)
        self.rnn_cell = LTCCellPaperExact(
            wiring=wiring,
            in_features=input_size,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon
        )
    
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
            input: Input tensor [batch, seq, features] or [seq, batch, features]
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
        else:
            h_state = hx
            if not is_batched:
                h_state = h_state.unsqueeze(0)
        
        # Process sequence
        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()
            
            h_out, h_state = self.rnn_cell(inputs, h_state, ts)
            
            if self.return_sequences:
                output_sequence.append(h_out)
        
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            h_state = h_state[0]
        
        return readout, h_state


# =============================================================================
# COMPARISON HELPER
# =============================================================================

def compare_implementations(
    batch_size: int = 2,
    seq_len: int = 5,
    input_size: int = 8,
    hidden_size: int = 16,
    device: str = 'cpu'
):
    """
    Compare paper-exact implementation with enhanced version.
    
    This function demonstrates the differences in computation.
    """
    print("=" * 80)
    print("LTC IMPLEMENTATION COMPARISON")
    print("=" * 80)
    
    # Create models
    ltc_paper = LTCPaperExact(input_size, hidden_size, ode_unfolds=6)
    
    # Generate random input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    print("\n1. Paper Exact Implementation:")
    print("   - Uses fused ODE solver (Equation 3)")
    print("   - Simple f = tanh(W_r·x + W_s·I + b)")
    print("   - Simple attractor A")
    
    out_paper, h_paper = ltc_paper(x)
    print(f"   Output shape: {out_paper.shape}")
    print(f"   Final hidden state shape: {h_paper.shape}")
    
    # Show components
    components = ltc_paper.rnn_cell.get_ode_components(
        x[:, 0], 
        torch.zeros(batch_size, hidden_size)
    )
    print(f"\n2. Paper Components:")
    print(f"   - tau shape: {components['tau'].shape}")
    print(f"   - f(·) shape: {components['f'].shape}")
    print(f"   - A shape: {components['A'].shape}")
    print(f"   - W_r shape: {components['W_r'].shape}")
    print(f"   - W_s shape: {components['W_s'].shape}")
    
    print("\n" + "=" * 80)
    print("Key Differences from ltc_cell.py:")
    print("=" * 80)
    print("1. Fused solver:  [x + Δt·f·A] / [1 + Δt(1/τ + f)]")
    print("   vs Euler:      x + Δt·[(−x + eff_input)/τ]")
    print("\n2. Simple f(·):   tanh(W_r·x + W_s·I + b)")
    print("   vs Complex:    Reversal potentials + CM gate")
    print("\n3. Simple A:      Learnable bias vector")
    print("   vs Biological: Weighted conductances")
    print("=" * 80)


if __name__ == "__main__":
    compare_implementations()
