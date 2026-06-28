"""
================================================================================
TRAJECTRON++ ENCODER — LTC/CfC DECODER — PFM ADJUSTMENT HYBRID MODEL
================================================================================

This module implements the proposed hybrid architecture that combines:

1. **Trajectron++ Encoder** (Graph-Based CVAE Encoding)
   - Node History Encoder: LSTM per agent encodes past trajectory
   - Edge Encoder: LSTM encodes joint neighbor-ego interaction history
   - Edge Influence Combiner: Additive attention or sum aggregation
   - Node Future Encoder: Bidirectional LSTM (training only, for q(z|x,y))
   - CVAE Latent Variable z: Discrete latent with Gumbel-Softmax

2. **LTC / CfC Decoder** (Replacing the original GRU + GMM)
   - Initialized with [z, x] (latent code + encoder context)
   - CfC cells for continuous-time autoregressive decoding
   - Kinematic integration with angular velocity (optional)

3. **PFM Adjustment** (Physics-Informed Post-Processing)
   - Goal attraction, prediction attraction, neighbor repulsion
   - Learnable per-agent force coefficients
   - Speed constraints for physical plausibility

================================================================================
SUB-MODULE LAYOUT
================================================================================

  models/trajectron_latent.py    — ModeKeys, DiscreteLatent
  models/trajectron_attention.py — AdditiveAttention
  models/trajectron_encoder.py   — TrajectronEncoder
  models/trajectron_pfm.py       — PotentialField
  models/trajectron_ltc_model.py — TrajectronLTC  (this file)

  utils/cfc_cell.py              — CfCCell, CfC  (existing)

================================================================================
REFERENCE
================================================================================

Original Trajectron++ paper:
  Salzmann, Ivanovic, Chakravarty, Pavone.
  "Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous
   Data", ECCV 2020.
  GitHub: https://github.com/StanfordASL/Trajectron-plus-plus

LTC / CfC papers:
  Hasani et al., "Liquid Time-Constant Networks", AAAI 2021.
  Hasani et al., "Closed-form Continuous-time Neural Networks", Nature ML 2022.

================================================================================
ARCHITECTURE DIAGRAM
================================================================================

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                             INPUT LAYER                                │
  │  history_neighbors: [B, A, ent, H, 2]    goals: [B, A, ent, 2]       │
  │  future (training):  [B, A, T, 2]                                     │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼
  ┌───────────────┐     ┌─────────────────┐
  │  Node History │     │  Edge Encoder   │
  │  Encoder      │     │  (per neighbor  │
  │  LSTM → h_n   │     │   type)         │
  └───────┬───────┘     │  LSTM → h_edge  │
          │             └────────┬────────┘
          │                      │
          │             ┌────────▼────────┐
          │             │ Edge Influence  │
          │             │ Combiner        │
          │             │ (attention/sum) │
          │             └────────┬────────┘
          │                      │
          └──────────┬───────────┘
                     │
                     ▼   x = concat(h_n, h_edge)
          ┌──────────────────────┐
          │     CVAE Encoder     │
          ├──────────────────────┤
          │ TRAIN:               │       ┌──────────────────┐
          │   y_e = BiLSTM(y)    │←──────│ Node Future Enc  │
          │   q(z|x,y) → z      │       │ (training only)  │
          │                      │       └──────────────────┘
          │ PREDICT:             │
          │   p(z|x) → z        │
          └──────────┬──────────┘
                     │
                     ▼  context = [z, x]
          ┌──────────────────────┐
          │   CfC / LTC Decoder  │
          ├──────────────────────┤
          │ h0 = Linear(context) │
          │ For t = 1..T:        │
          │   input_t = embed(   │
          │     prev_pred)       │
          │   h_t = CfC(in, h)   │
          │   out_t = Linear(h)  │
          │   pred_t = integrate │
          └──────────┬──────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   PFM Adjustment     │
          ├──────────────────────┤
          │ F_goal + F_pred      │
          │      + F_rep         │
          │ Speed constraints    │
          └──────────┬──────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │       OUTPUT         │
          │ adjusted_preds       │
          │ decoded_preds        │
          │ kl_loss, coeff stats │
          └──────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ---------------------------------------------------------------------------
# Path setup (works both when run directly and when imported)
# ---------------------------------------------------------------------------
try:
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    _BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

if _BASE_DIR not in sys.path:
    sys.path.append(_BASE_DIR)

# ---------------------------------------------------------------------------
# Sub-module imports  (each class lives in its own focused file)
# ---------------------------------------------------------------------------
from models.trajectron_latent    import ModeKeys, DiscreteLatent       # noqa: F401
from models.trajectron_attention import AdditiveAttention              # noqa: F401
from models.trajectron_encoder   import TrajectronEncoder              # noqa: F401
from models.trajectron_pfm       import PotentialField                 # noqa: F401
from utils.cfc_cell              import CfCCell, CfC                   # noqa: F401


# =============================================================================
# MAIN MODEL: TRAJECTRON++ ENCODER → LTC/CfC DECODER → PFM
# =============================================================================
class TrajectronLTC(nn.Module):
    """
    Trajectron++ Encoder + LTC/CfC Decoder + PFM Adjustment.

    This is the full hybrid architecture that:
    1. Uses Trajectron++ style graph-based CVAE encoding
    2. Replaces the original GRU+GMM decoder with CfC/LTC cells
    3. Applies PFM post-processing for physics-informed adjustment

    Args:
        state_dim (int):                Position dimension (default: 2)
        pred_state_dim (int):           Prediction dimension (default: 2)
        enc_rnn_dim_history (int):      Node history encoder hidden dim
        enc_rnn_dim_edge (int):         Edge encoder hidden dim
        enc_rnn_dim_future (int):       Future encoder hidden dim
        edge_influence_combine (str):   "attention" or "sum"
        dec_rnn_dim (int):              Decoder CfC hidden dimension
        N (int):                        Latent categorical variables
        K (int):                        Categories per latent variable
        prediction_horizon (int):       Number of future timesteps
        cfc_mode (str):                 "default", "pure", or "no_gate"
        cfc_backbone_units (int):       CfC backbone MLP hidden size
        cfc_backbone_layers (int):      CfC backbone MLP depth
        cfc_backbone_activation (str):  Activation function
        cfc_backbone_dropout (float):   Dropout rate
        mixed_memory (bool):            Augment CfC with LSTM memory
        use_angular_velocity (bool):    Kinematic integration mode
        target_avg_speed (float):       Expected agent speed
        speed_tolerance (float):        Allowed speed deviation fraction
        dt (float):                     Integration timestep
        num_agents (int):               Max agents for PFM embedding table
        pfm_k_init (float):             Initial PFM coefficient value
        pfm_repulsion_radius (float):   Neighbour repulsion radius
        dropout (float):                General dropout rate
        p_z_x_MLP_dims (int):          Hidden dim for prior MLP
        q_z_xy_MLP_dims (int):         Hidden dim for posterior MLP
    """

    def __init__(
        self,
        # Encoder
        state_dim: int               = 2,
        pred_state_dim: int          = 2,
        enc_rnn_dim_history: int     = 32,
        enc_rnn_dim_edge: int        = 32,
        enc_rnn_dim_future: int      = 32,
        edge_influence_combine: str  = "attention",
        # Decoder
        dec_rnn_dim: int             = 128,
        # Latent
        N: int                       = 1,
        K: int                       = 25,
        # Prediction
        prediction_horizon: int      = 12,
        # CfC
        cfc_mode: str                = "default",
        cfc_backbone_units: int      = 128,
        cfc_backbone_layers: int     = 1,
        cfc_backbone_activation: str = "lecun_tanh",
        cfc_backbone_dropout: float  = 0.0,
        mixed_memory: bool           = True,
        # Kinematic
        use_angular_velocity: bool   = True,
        target_avg_speed: float      = 4.087,
        speed_tolerance: float       = 0.15,
        dt: float                    = 0.1,
        # PFM
        num_agents: int              = 1000,
        pfm_k_init: float            = 1.0,
        pfm_repulsion_radius: float  = 0.5,
        # General
        dropout: float               = 0.0,
        # CVAE
        p_z_x_MLP_dims: int          = 32,
        q_z_xy_MLP_dims: int         = 32,
    ):
        super().__init__()

        self.state_dim            = state_dim
        self.pred_state_dim       = pred_state_dim
        self.prediction_horizon   = prediction_horizon
        self.use_angular_velocity = use_angular_velocity
        self.dt                   = dt
        self.dec_rnn_dim          = dec_rnn_dim
        self.mixed_memory         = mixed_memory

        # Speed constraints
        self.target_avg_speed = target_avg_speed
        self.speed_tolerance  = speed_tolerance
        self.min_speed        = target_avg_speed * (1 - speed_tolerance)
        self.max_speed        = target_avg_speed * (1 + speed_tolerance)

        # =====================================================================
        # ENCODER  (Trajectron++ style, defined in trajectron_encoder.py)
        # =====================================================================
        self.encoder = TrajectronEncoder(
            state_dim=state_dim,
            pred_state_dim=pred_state_dim,
            enc_rnn_dim_history=enc_rnn_dim_history,
            enc_rnn_dim_edge=enc_rnn_dim_edge,
            enc_rnn_dim_future=enc_rnn_dim_future,
            edge_influence_combine=edge_influence_combine,
            N=N, K=K,
            p_z_x_MLP_dims=p_z_x_MLP_dims,
            q_z_xy_MLP_dims=q_z_xy_MLP_dims,
            dropout=dropout,
        )

        x_size = self.encoder.x_size
        z_size = self.encoder.z_size

        # =====================================================================
        # DECODER INPUT PROJECTION
        # Maps [z, x] → initial decoder hidden / cell states
        # =====================================================================
        self.decoder_initial_h = nn.Linear(z_size + x_size, dec_rnn_dim)

        if mixed_memory:
            self.decoder_initial_c = nn.Linear(z_size + x_size, dec_rnn_dim)

        # Current state → action seed
        self.state_action = nn.Linear(state_dim, pred_state_dim)

        # =====================================================================
        # CfC DECODER CELL  (from utils/cfc_cell.py)
        # Decoder input: concat([z, x], a_t)
        # =====================================================================
        self.decoder_input_dim = z_size + x_size + pred_state_dim

        self.decoder_cell = CfCCell(
            input_size=self.decoder_input_dim,
            hidden_size=dec_rnn_dim,
            mode=cfc_mode,
            backbone_activation=cfc_backbone_activation,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            backbone_dropout=cfc_backbone_dropout,
        )

        if mixed_memory:
            self.decoder_lstm_cell = nn.LSTMCell(self.decoder_input_dim,
                                                  dec_rnn_dim)

        # =====================================================================
        # OUTPUT HEAD
        # =====================================================================
        out_dim = 2 if use_angular_velocity else pred_state_dim
        self.output_head = nn.Linear(dec_rnn_dim, out_dim)

        # =====================================================================
        # COEFFICIENT PROJECTOR  (encoder context → PFM coefficients)
        # =====================================================================
        self.coeff_projector = nn.Linear(x_size, 3)

        # =====================================================================
        # POTENTIAL FIELD MODULE  (defined in trajectron_pfm.py)
        # =====================================================================
        self.pfm = PotentialField(
            num_agents=num_agents,
            k_init=pfm_k_init,
            repulsion_radius=pfm_repulsion_radius,
        )

    # =========================================================================
    # KINEMATIC FEATURE EXTRACTION
    # =========================================================================
    def compute_velocity_features(self, history: torch.Tensor):
        """
        Extract v_avg and last heading angle from position history.

        Args:
            history: [batch, H, 2]
        Returns:
            v_avg:      [batch]
            theta_last: [batch]
        """
        B, H, _ = history.shape
        device   = history.device

        displacements = history[:, 1:] - history[:, :-1]      # [B, H-1, 2]
        distances     = torch.norm(displacements, dim=-1)       # [B, H-1]
        velocities    = distances / self.dt

        dx    = displacements[:, :, 0]
        dy    = displacements[:, :, 1]
        theta = torch.atan2(dy, dx)

        valid_mask = velocities > 0.01
        v_avg = torch.where(
            valid_mask.any(dim=1),
            velocities.sum(dim=1) / (valid_mask.sum(dim=1).float() + 1e-6),
            torch.full((B,), self.target_avg_speed, device=device),
        )

        theta_last = theta[:, -1] if H > 1 else torch.zeros(B, device=device)
        return v_avg, theta_last

    # =========================================================================
    # AUTOREGRESSIVE CfC DECODER
    # =========================================================================
    def decode_autoregressive(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        n_s_t0: torch.Tensor,
        last_pos: torch.Tensor,
        v_avg: torch.Tensor      = None,
        theta_last: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        CfC autoregressive decoder.

        Replaces the original Trajectron++ GRU + GMM decoder with:
          - CfC cell for continuous-time dynamics
          - Direct position prediction (instead of GMM)
          - Optional kinematic integration with angular velocity

        Args:
            x:          [batch, x_size]    encoder context
            z:          [batch, z_size]    latent sample
            n_s_t0:     [batch, state_dim] current state
            last_pos:   [batch, 2]         last known position
            v_avg:      [batch]            average speed (angular-vel mode)
            theta_last: [batch]            last heading  (angular-vel mode)
        Returns:
            predictions: [batch, T, 2]
        """
        B      = x.shape[0]
        device = x.device
        T      = self.prediction_horizon

        zx      = torch.cat([z, x], dim=-1)
        h_state = torch.tanh(self.decoder_initial_h(zx))

        if self.mixed_memory:
            c_state = torch.tanh(self.decoder_initial_c(zx))

        a_t           = self.state_action(n_s_t0)
        predictions   = []
        current_pos   = last_pos.clone()
        current_theta = (theta_last.clone() if theta_last is not None
                         else torch.zeros(B, device=device))

        for t in range(T):
            dec_input = torch.cat([zx, a_t], dim=-1)

            if self.mixed_memory:
                h_state, c_state = self.decoder_lstm_cell(dec_input,
                                                           (h_state, c_state))

            h_out, h_state = self.decoder_cell(dec_input, h_state, ts=1.0)
            out            = self.output_head(h_out)

            if self.use_angular_velocity and v_avg is not None:
                delta_v     = out[:, 0]
                delta_omega = out[:, 1]

                v_pred        = torch.clamp(v_avg + delta_v,
                                            self.min_speed, self.max_speed)
                current_theta = current_theta + delta_omega * self.dt
                current_theta = torch.atan2(torch.sin(current_theta),
                                            torch.cos(current_theta))

                dx          = v_pred * torch.cos(current_theta) * self.dt
                dy          = v_pred * torch.sin(current_theta) * self.dt
                current_pos = current_pos + torch.stack([dx, dy], dim=-1)
            else:
                current_pos = current_pos + out

            predictions.append(current_pos.clone())
            # Teacher-less: next action from current output
            a_t = out.detach() if not self.training else out

        return torch.stack(predictions, dim=1)   # [batch, T, 2]

    # =========================================================================
    # FORWARD
    # =========================================================================
    def forward(
        self,
        history_neighbors: torch.Tensor,
        goal: torch.Tensor,
        future: torch.Tensor     = None,
        mode: ModeKeys           = None,
        num_samples: int         = 1,
    ):
        """
        Full forward pass: Trajectron++ Encode → CfC Decode → PFM Adjust.

        Args:
            history_neighbors: [B, A, ent, H, 2]
                B=batch, A=agents, ent=entities (1 ego + neighbors), H=history
            goal:        [B, A, 2] or [B, A, ent, 2]
            future:      [B, A, T, 2]  ego future trajectory (training only)
            mode:        ModeKeys or None (auto-detected from future)
            num_samples: number of latent samples

        Returns:
            adjusted_preds: [B, A, ent, T, 2]  physics-corrected predictions
            decoded_preds:  [B, A, ent, T, 2]  raw neural predictions
            kl_loss:        scalar tensor or None
            coeff_mean:     scalar
            coeff_var:      scalar
        """
        B, A, ent, H, D = history_neighbors.shape
        device = history_neighbors.device
        T      = self.prediction_horizon

        # Auto-detect mode
        if mode is None:
            mode = ModeKeys.TRAIN if future is not None else ModeKeys.PREDICT

        # ── Goal shape normalisation ─────────────────────────────────────────
        if goal.dim() == 3 and goal.shape == (B, A, D):
            goal_expanded = goal.unsqueeze(2).expand(B, A, ent, D).contiguous()
        elif goal.dim() == 4 and goal.shape == (B, A, ent, D):
            goal_expanded = goal
        else:
            goal_expanded = goal.unsqueeze(2).expand(B, A, ent, D).contiguous()

        # ── Split ego / neighbour histories ──────────────────────────────────
        ego_history   = history_neighbors[:, :, 0, :, :]   # [B, A, H, D]
        nbr_histories = (history_neighbors[:, :, 1:, :, :]
                         if ent > 1
                         else torch.zeros(B, A, 0, H, D, device=device))

        ego_hist_flat = ego_history.reshape(B * A, H, D)
        nbr_hist_flat = nbr_histories.reshape(B * A, max(ent - 1, 0), H, D)

        ego_future_flat = (future.reshape(B * A, T, D)
                           if future is not None else None)

        # ── Encode ───────────────────────────────────────────────────────────
        x, z, kl_loss = self.encoder(
            ego_history=ego_hist_flat,
            neighbor_histories=nbr_hist_flat,
            ego_future=ego_future_flat,
            mode=mode,
            num_samples=num_samples,
        )
        # x: [B*A, x_size]   z: [num_samples, B*A, z_size]

        # ── PFM coefficient projection ─────────────────────────────────────────
        coeffs     = self.coeff_projector(x).reshape(B, A, 3)
        coeffs_ent = coeffs.unsqueeze(2).expand(B, A, ent, 3).contiguous()

        # ── Kinematic features ────────────────────────────────────────────────
        if self.use_angular_velocity:
            v_avg, theta_last = self.compute_velocity_features(ego_hist_flat)
        else:
            v_avg, theta_last = None, None

        all_adjusted_preds = []
        all_decoded_preds = []
        all_coeff_mean = []
        all_coeff_var = []

        # ── Loop over latent samples (Best-of-N / Variety generation) ─────────
        for k in range(num_samples):
            z_sample      = z[k]                         # [B*A, z_size]
            last_pos_flat = ego_hist_flat[:, -1, :]      # [B*A, D]

            ego_preds_flat = self.decode_autoregressive(
                x=x, z=z_sample, n_s_t0=last_pos_flat,
                last_pos=last_pos_flat,
                v_avg=v_avg, theta_last=theta_last,
            )                                             # [B*A, T, D]

            ego_preds = ego_preds_flat.reshape(B, A, T, D)

            # ── Decode all entities ───────────────────────────────────────────────
            # Ego: neural decode.  Neighbours: linear extrapolation (auxiliary).
            k_decoded_preds = torch.zeros(B, A, ent, T, D, device=device)
            k_decoded_preds[:, :, 0] = ego_preds

            for e_idx in range(1, ent):
                nbr_last = history_neighbors[:, :, e_idx, -1, :]
                nbr_vel  = (history_neighbors[:, :, e_idx, -1, :]
                            - history_neighbors[:, :, e_idx, -2, :]
                            if H > 1
                            else torch.zeros_like(nbr_last))
                for t in range(T):
                    k_decoded_preds[:, :, e_idx, t] = nbr_last + nbr_vel * (t + 1)

            # ── PFM correction loop ───────────────────────────────────────────────
            k_adjusted_preds = torch.zeros_like(k_decoded_preds)
            current_pos    = history_neighbors[:, :, :, -1, :].clone()  # [B,A,ent,D]

            coeff_list = []
            for t in range(T):
                for idx in range(ent):
                    pos      = current_pos[:, :, idx]
                    pred     = k_decoded_preds[:, :, idx, t]
                    goal_vec = goal_expanded[:, :, idx]
                    coeff    = coeffs_ent[:, :, idx]

                    neighbor_idxs = [i for i in range(ent) if i != idx]
                    if neighbor_idxs:
                        neighbors_pos = torch.stack(
                            [current_pos[:, :, j] for j in neighbor_idxs], dim=2
                        )
                    else:
                        neighbors_pos = torch.empty(B, A, 0, D, device=device)

                    force, coeff_upd = self.pfm(pos, pred, neighbors_pos,
                                                goal_vec, coeff)

                    new_pos = (pos + force if t == 0
                               else k_adjusted_preds[:, :, idx, t - 1] + force)

                    # Speed constraint clipping
                    if t > 0:
                        prev    = k_adjusted_preds[:, :, idx, t - 1]
                        disp    = new_pos - prev
                        speed   = torch.norm(disp, dim=-1, keepdim=True)
                        clipped = torch.clamp(speed,
                                             self.min_speed * self.dt,
                                             self.max_speed * self.dt)
                        new_pos = prev + disp / (speed + 1e-8) * clipped

                    k_adjusted_preds[:, :, idx, t] = new_pos

                    if idx == 0:
                        coeff_list.append(coeff_upd)

                current_pos = k_adjusted_preds[:, :, :, t].clone()

            # ── Coefficient statistics ─────────────────────────────────────────────
            if coeff_list:
                coeff_stack = torch.stack(coeff_list)
                c_mean  = coeff_stack.mean()
                c_var   = coeff_stack.var(unbiased=False)
            else:
                c_mean  = torch.tensor(0.0, device=device)
                c_var   = torch.tensor(0.0, device=device)
                
            all_adjusted_preds.append(k_adjusted_preds)
            all_decoded_preds.append(k_decoded_preds)
            all_coeff_mean.append(c_mean)
            all_coeff_var.append(c_var)

        # Stack over num_samples
        adjusted_preds = torch.stack(all_adjusted_preds) # [num_samples, B, A, ent, T, D]
        decoded_preds  = torch.stack(all_decoded_preds)  # [num_samples, B, A, ent, T, D]
        coeff_mean     = torch.stack(all_coeff_mean).mean()
        coeff_var      = torch.stack(all_coeff_var).mean()

        # If num_samples == 1, remove the sample dimension for backward compat,
        # unless specifically requesting multiple.
        if num_samples == 1:
            adjusted_preds = adjusted_preds.squeeze(0)
            decoded_preds  = decoded_preds.squeeze(0)

        return adjusted_preds, decoded_preds, kl_loss, coeff_mean, coeff_var

    # =========================================================================
    # ELBO TRAINING LOSS
    # =========================================================================
    def train_loss(
        self,
        history_neighbors: torch.Tensor,
        goal: torch.Tensor,
        future: torch.Tensor,
        all_futures: torch.Tensor = None,
        kl_weight: float          = 1.0,
        coeff_reg_weight: float   = 0.01,
        num_samples: int          = 10,   # Set Best-of-N K for variety loss
    ):
        """
        Compute ELBO-based training loss with Best-of-N (Variety Loss).

            Loss = NLL (Best-of-N) + kl_weight * KL(q||p) + coeff_reg

        Args:
            history_neighbors: [B, A, ent, H, 2]
            goal:              [B, A, 2]
            future:            [B, A, T, 2]  ground-truth ego future
            all_futures:       [B, A, ent, T, 2]  all-entity futures (optional)
            kl_weight:         KL annealing weight
            coeff_reg_weight:  coefficient regularization weight
            num_samples:       Number of samples for Best-of-N loss

        Returns:
            total_loss: scalar
            loss_dict:  dict of individual loss components
        """
        adjusted_preds, decoded_preds, kl_loss, coeff_mean, coeff_var = self.forward(
            history_neighbors=history_neighbors,
            goal=goal,
            future=future,
            mode=ModeKeys.TRAIN,
            num_samples=num_samples,
        )

        # NLL: Best-of-N (Variety Loss) on ego trajectory
        # adjusted_preds is [num_samples, B, A, ent, T, D]
        ego_adjusted = adjusted_preds[:, :, :, 0]    # [num_samples, B, A, T, D]
        ego_decoded  = decoded_preds[:, :, :, 0]     # [num_samples, B, A, T, D]

        # Calculate MSE for all samples
        # future is [B, A, T, D] -> broadcast to [num_samples, B, A, T, D]
        nll_adjusted_all = torch.norm(ego_adjusted - future.unsqueeze(0), dim=-1).sum(dim=-1) # [num_samples, B, A]
        nll_decoded_all  = torch.norm(ego_decoded - future.unsqueeze(0), dim=-1).sum(dim=-1)  # [num_samples, B, A]
        
        # Take the minimum error across the num_samples dimension (Best-of-N)
        best_nll_adjusted, _ = nll_adjusted_all.min(dim=0) # [B, A]
        best_nll_decoded, _  = nll_decoded_all.min(dim=0)  # [B, A]
        
        nll_adjusted = best_nll_adjusted.mean()
        nll_decoded  = best_nll_decoded.mean()
        nll_loss     = 0.5 * nll_adjusted + 0.5 * nll_decoded

        # Entity loss (if all futures provided)
        if all_futures is not None:
            # Also apply Best-of-N to entity loss
            entity_loss_all = torch.norm(adjusted_preds - all_futures.unsqueeze(0), dim=-1).sum(dim=(3, 4)) # [num_samples, B, A]
            best_entity_loss, _ = entity_loss_all.min(dim=0)
            entity_loss = best_entity_loss.mean()
        else:
            entity_loss = torch.tensor(0.0, device=future.device)

        # KL
        if kl_loss is None:
            kl_loss = torch.tensor(0.0, device=future.device)

        # Coefficient regularization (variance penalisation)
        coeff_reg = coeff_var

        total_loss = (nll_loss
                      + kl_weight * kl_loss
                      + 0.1 * entity_loss
                      + coeff_reg_weight * coeff_reg)

        loss_dict = {
            'total_loss':   total_loss.item(),
            'nll_adjusted': nll_adjusted.item(),
            'nll_decoded':  nll_decoded.item(),
            'kl_loss':      (kl_loss.item()
                             if isinstance(kl_loss, torch.Tensor) else kl_loss),
            'entity_loss':  entity_loss.item(),
            'coeff_mean':   coeff_mean.item(),
            'coeff_var':    coeff_var.item(),
        }

        return total_loss, loss_dict

    # =========================================================================
    # PREDICTION (inference)
    # =========================================================================
    def predict(
        self,
        history_neighbors: torch.Tensor,
        goal: torch.Tensor,
        num_samples: int         = 20,
    ):
        """
        Generate predictions without gradients.

        Args:
            history_neighbors: [B, A, ent, H, 2]
            goal:              [B, A, 2]
            num_samples:       number of latent samples

        Returns:
            adjusted_preds: [B, A, ent, T, 2]
            decoded_preds:  [B, A, ent, T, 2]
        """
        with torch.no_grad():
            adjusted, decoded, _, _, _ = self.forward(
                history_neighbors=history_neighbors,
                goal=goal,
                future=None,
                mode=ModeKeys.PREDICT,
                num_samples=num_samples,
            )
        return adjusted, decoded


# =============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================
TrajectronPlusPlusLTC = TrajectronLTC
TrajectronCfCPFM      = TrajectronLTC
