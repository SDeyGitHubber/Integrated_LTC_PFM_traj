"""
================================================================================
TRAJECTRON++ — MODE KEYS & DISCRETE LATENT VARIABLE
================================================================================

Contains:
  - ModeKeys: Train / Eval / Predict mode enumeration
  - DiscreteLatent: Gumbel-Softmax CVAE latent variable

Used by:
  TrajectronEncoder  (models/trajectron_encoder.py)
  TrajectronLTC      (models/trajectron_ltc_model.py)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


# =============================================================================
# MODE KEYS (matching Trajectron++ convention)
# =============================================================================
class ModeKeys(Enum):
    TRAIN   = "train"
    EVAL    = "eval"
    PREDICT = "predict"


# =============================================================================
# DISCRETE LATENT VARIABLE (Gumbel-Softmax CVAE)
# =============================================================================
class DiscreteLatent(nn.Module):
    """
    Discrete latent variable for Trajectron++ CVAE.

    Uses Gumbel-Softmax for differentiable sampling from a categorical
    distribution during training, and argmax during evaluation/prediction.

    The latent space is factored into N independent categorical variables,
    each with K categories, giving z_dim = N * K.

    Args:
        N (int):        Number of independent categorical variables (default: 1)
        K (int):        Number of categories per variable (default: 25)
        kl_min (float): Minimum KL divergence clipping value
    """

    def __init__(self, N: int = 1, K: int = 25, kl_min: float = 0.07):
        super().__init__()
        self.N      = N
        self.K      = K
        self.z_dim  = N * K
        self.kl_min = kl_min

        # Temperature for Gumbel-Softmax (annealed during training externally)
        self.temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Optional logit clipping
        self.z_logit_clip = None

        # Distribution placeholders (set by encoder before sampling)
        self.p_dist = None   # Prior  p(z|x)
        self.q_dist = None   # Posterior q(z|x,y)

    def dist_from_h(self, h: torch.Tensor, mode: ModeKeys) -> torch.Tensor:
        """
        Reshape flat logits into [batch, N, K] distribution tensor.

        Args:
            h:    [batch, N*K] raw logits
            mode: Operating mode (for optional clipping during training)
        Returns:
            logits: [batch, N, K]
        """
        logits = h.reshape(-1, self.N, self.K)

        if self.z_logit_clip is not None and mode == ModeKeys.TRAIN:
            logits = torch.clamp(logits, -self.z_logit_clip, self.z_logit_clip)

        return logits

    def sample_q(self, num_samples: int, mode: ModeKeys) -> torch.Tensor:
        """
        Sample from posterior q(z|x,y).

        Args:
            num_samples: Number of latent samples to draw
            mode:        Operating mode (soft in TRAIN, hard in EVAL)
        Returns:
            z: [num_samples, batch, N*K] one-hot style samples
        """
        bs   = self.q_dist.shape[0]
        hard = (mode != ModeKeys.TRAIN)

        z_NK = F.gumbel_softmax(
            self.q_dist.reshape(-1, self.K),
            tau=self.temp.item(),
            hard=hard,
        ).reshape(bs, self.N * self.K)

        return z_NK.unsqueeze(0).expand(num_samples, -1, -1)

    def sample_p(self, num_samples: int, mode: ModeKeys,
                 most_likely_z: bool = False,
                 full_dist: bool = True,
                 all_z_sep: bool = False):
        """
        Sample from prior p(z|x).

        Args:
            num_samples:   Number of samples
            mode:          Operating mode
            most_likely_z: If True, use argmax (single most-likely z)
            full_dist:     If True, enumerate all K latent modes
            all_z_sep:     Kept for API compatibility (unused)
        Returns:
            (z, num_samples_out, num_components)
        """
        bs = self.p_dist.shape[0]

        if most_likely_z:
            eye      = torch.eye(self.K, device=self.p_dist.device)
            argmax   = torch.argmax(self.p_dist, dim=-1)   # [bs, N]
            z        = eye[argmax].reshape(bs, self.N * self.K)
            return z.unsqueeze(0), 1, 1

        if full_dist:
            eye   = torch.eye(self.K, device=self.p_dist.device)
            all_z = [
                eye[k_idx].unsqueeze(0).unsqueeze(0)
                          .expand(1, bs, self.N)
                          .reshape(1, bs, self.N * self.K)
                for k_idx in range(self.K)
            ]
            z = torch.cat(all_z, dim=0)   # [K, bs, N*K]
            return z, self.K, self.K

        # Random sampling
        z_NK = F.gumbel_softmax(
            self.p_dist.reshape(-1, self.K),
            tau=self.temp.item(),
            hard=True,
        ).reshape(bs, self.N * self.K)
        z = z_NK.unsqueeze(0).expand(num_samples, -1, -1)
        return z, num_samples, 1

    def kl_q_p(self) -> torch.Tensor:
        """
        Analytical KL(q || p) for categorical distributions.

            KL = Σ q(z) * [log q(z) − log p(z)]

        Returns:
            kl: Scalar (mean over batch, sum over N and K)
        """
        q_probs = F.softmax(self.q_dist, dim=-1)
        p_probs = F.softmax(self.p_dist, dim=-1)

        kl = (q_probs * (torch.log(q_probs + 1e-8)
                       - torch.log(p_probs + 1e-8))).sum(-1)  # [bs, N]
        kl = kl.sum(-1)                                        # [bs]
        kl = torch.clamp(kl, min=self.kl_min)

        return kl.mean()
