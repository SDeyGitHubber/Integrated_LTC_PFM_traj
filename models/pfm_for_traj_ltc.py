"""
================================================================================
TRAJECTRON++ — POTENTIAL FIELD MODULE (PFM)
================================================================================

Physics-inspired social force module applied as a post-processing step
on top of the neural decoder output.

Forces computed per entity per timestep:
  F_goal = k1 * (goal  − pos)               — Goal attraction
  F_pred = k2 * (pred  − pos)               — Prediction attraction
  F_rep  = Σ  kr * (pos − nbr) / ||d||²     — Neighbor repulsion

Each agent gets its own learnable force coefficients projected from the
encoder context (see TrajectronLTC.coeff_projector).

Used by:
  models.trajectron_ltc_model → TrajectronLTC
================================================================================
"""

import torch
import torch.nn as nn


class PotentialField(nn.Module):
    """
    Potential Field Module (PFM) for physics-informed trajectory adjustment.

    Args:
        num_agents (int):         Embedding table size (kept for compat,
                                  TrajectronLTC always projects from context)
        k_init (float):           Initial value for embedding weights
        repulsion_radius (float): Distance threshold for neighbour repulsion
    """

    def __init__(
        self,
        num_agents: int        = 1000,
        k_init: float          = 1.0,
        repulsion_radius: float = 0.5,
    ):
        super().__init__()
        self.repulsion_radius = repulsion_radius

        # Legacy embedding — retained for checkpoint compatibility.
        # In TrajectronLTC, coefficients are projected from the encoder context
        # via 'coeff_projector' and passed in at call time.
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)

    def forward(
        self,
        pos: torch.Tensor,
        predicted: torch.Tensor,
        neighbors: torch.Tensor,
        goal: torch.Tensor,
        coeffs: torch.Tensor,
    ):
        """
        Compute the APF-based force vector for each agent.

        Args:
            pos:       [B, A, D]    current agent positions
            predicted: [B, A, D]   predicted next positions
            neighbors: [B, A, N, D] neighbouring agents' positions
            goal:      [B, A, D]   goal positions
            coeffs:    [B, A, 3]   force coefficients (k1, k2, kr)

        Returns:
            total_force: [B, A, D]
            coeffs:      [B, A, 3]  (unchanged, returned for logging)
        """
        k1 = coeffs[..., 0:1]   # goal attraction
        k2 = coeffs[..., 1:2]   # prediction attraction
        kr = coeffs[..., 2:3]   # neighbour repulsion

        # ── Attractive forces ────────────────────────────────────────────────
        F_goal = k1 * (goal      - pos)    # [B, A, D]
        F_pred = k2 * (predicted - pos)    # [B, A, D]

        # ── Repulsive force ──────────────────────────────────────────────────
        if neighbors.size(-2) == 0:
            F_rep = torch.zeros_like(pos)
        else:
            diffs     = pos.unsqueeze(-2) - neighbors              # [B, A, N, D]
            dists     = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
            safe_dist = torch.clamp(dists, min=1e-6)
            mask      = ((dists < self.repulsion_radius)
                         & (dists > 1e-5)).float()
            F_rep = (kr.unsqueeze(-2) * diffs / safe_dist.pow(2) * mask).sum(-2)

        return F_goal + F_pred + F_rep, coeffs