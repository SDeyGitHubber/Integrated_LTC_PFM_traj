"""
================================================================================
TRAJECTRON++ — GRAPH-BASED CVAE ENCODER
================================================================================

Implements the full Trajectron++ encoder which builds a spatio-temporal
representation from:
  1. Each agent's past trajectory  (node history LSTM)
  2. Inter-agent interactions      (edge LSTM per neighbor)

Then uses a CVAE framework to produce a discrete latent code z:
  • p(z|x)    — prior  sampled at prediction time
  • q(z|x,y)  — posterior  used during training (teacher forcing)

Imports:
  models.trajectron_latent   → ModeKeys, DiscreteLatent
  models.trajectron_attention → AdditiveAttention

Used by:
  models.trajectron_ltc_model → TrajectronLTC
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trajectron_latent    import ModeKeys, DiscreteLatent
from models.trajectron_attention import AdditiveAttention


class TrajectronEncoder(nn.Module):
    """
    Trajectron++ style encoder for multi-agent trajectory prediction.

    Args:
        state_dim (int):               Input state dimension (default: 2 for x,y)
        pred_state_dim (int):          Prediction state dimension (default: 2)
        enc_rnn_dim_history (int):     Node history encoder hidden size
        enc_rnn_dim_edge (int):        Edge encoder hidden size
        enc_rnn_dim_future (int):      Future encoder hidden size
        edge_influence_combine (str):  "attention" | "sum" | "mean"
        N (int):                       Number of latent categorical variables
        K (int):                       Number of categories per variable
        p_z_x_MLP_dims (int | None):  MLP hidden size for p(z|x) (None = skip)
        q_z_xy_MLP_dims (int | None): MLP hidden size for q(z|x,y)
        dropout (float):               Dropout probability
    """

    def __init__(
        self,
        state_dim: int               = 2,
        pred_state_dim: int          = 2,
        enc_rnn_dim_history: int     = 32,
        enc_rnn_dim_edge: int        = 32,
        enc_rnn_dim_future: int      = 32,
        edge_influence_combine: str  = "attention",
        N: int                       = 1,
        K: int                       = 25,
        p_z_x_MLP_dims: int          = 32,
        q_z_xy_MLP_dims: int         = 32,
        dropout: float               = 0.0,
    ):
        super().__init__()

        self.state_dim              = state_dim
        self.pred_state_dim         = pred_state_dim
        self.enc_rnn_dim_history    = enc_rnn_dim_history
        self.enc_rnn_dim_edge       = enc_rnn_dim_edge
        self.enc_rnn_dim_future     = enc_rnn_dim_future
        self.edge_influence_combine = edge_influence_combine
        self.dropout                = dropout

        # =====================================================================
        # NODE HISTORY ENCODER
        # LSTM processes each agent's past trajectory
        # Input:  [batch, H, state_dim]
        # Output: final hidden state [batch, enc_rnn_dim_history]
        # =====================================================================
        self.node_history_encoder = nn.LSTM(
            input_size=state_dim,
            hidden_size=enc_rnn_dim_history,
            batch_first=True,
        )

        # =====================================================================
        # EDGE ENCODER
        # LSTM processes concatenated [neighbor_state, ego_state] per timestep
        # Input:  [batch, H, 2 * state_dim]
        # Output: final hidden state [batch, enc_rnn_dim_edge]
        # =====================================================================
        self.edge_encoder = nn.LSTM(
            input_size=state_dim + state_dim,
            hidden_size=enc_rnn_dim_edge,
            batch_first=True,
        )

        # =====================================================================
        # EDGE INFLUENCE COMBINER
        # Aggregates edge encodings from all neighbors into one vector
        # =====================================================================
        if edge_influence_combine == "attention":
            self.edge_influence_encoder = AdditiveAttention(
                encoder_dim=enc_rnn_dim_edge,
                decoder_dim=enc_rnn_dim_history,
            )
        # sum / mean require no learnable module
        self.eie_output_dims = enc_rnn_dim_edge

        # =====================================================================
        # ENCODER CONTEXT DIMENSION
        #   x = concat(node_history_enc, edge_influence_enc)
        # =====================================================================
        self.x_size = enc_rnn_dim_history + self.eie_output_dims

        # =====================================================================
        # NODE FUTURE ENCODER  (training only — computes posterior q(z|x,y))
        # Bidirectional LSTM on ground-truth future trajectory
        # Output size: 4 * enc_rnn_dim_future  (h_fwd, h_bwd, c_fwd, c_bwd)
        # =====================================================================
        self.node_future_encoder = nn.LSTM(
            input_size=pred_state_dim,
            hidden_size=enc_rnn_dim_future,
            bidirectional=True,
            batch_first=True,
        )
        self.future_encoder_initial_h = nn.Linear(state_dim, enc_rnn_dim_future)
        self.future_encoder_initial_c = nn.Linear(state_dim, enc_rnn_dim_future)
        self.future_enc_size = 4 * enc_rnn_dim_future

        # =====================================================================
        # DISCRETE LATENT VARIABLE
        # =====================================================================
        self.latent = DiscreteLatent(N=N, K=K)
        self.z_size = N * K

        # =====================================================================
        # p(z|x)  — Prior network
        # =====================================================================
        if p_z_x_MLP_dims is not None:
            self.p_z_x_mlp = nn.Sequential(
                nn.Linear(self.x_size, p_z_x_MLP_dims),
                nn.ReLU(),
            )
            self.hx_to_z = nn.Linear(p_z_x_MLP_dims, self.z_size)
        else:
            self.p_z_x_mlp = None
            self.hx_to_z   = nn.Linear(self.x_size, self.z_size)

        # =====================================================================
        # q(z|x,y)  — Posterior network (training only)
        # =====================================================================
        if q_z_xy_MLP_dims is not None:
            self.q_z_xy_mlp = nn.Sequential(
                nn.Linear(self.x_size + self.future_enc_size, q_z_xy_MLP_dims),
                nn.ReLU(),
            )
            self.hxy_to_z = nn.Linear(q_z_xy_MLP_dims, self.z_size)
        else:
            self.q_z_xy_mlp = None
            self.hxy_to_z   = nn.Linear(
                self.x_size + self.future_enc_size, self.z_size
            )

    # =========================================================================
    # ENCODE NODE HISTORY
    # =========================================================================
    def encode_node_history(self, node_history: torch.Tensor) -> torch.Tensor:
        """
        Encode each agent's past trajectory with LSTM.

        Args:
            node_history: [batch, H, state_dim]
        Returns:
            encoded: [batch, enc_rnn_dim_history]
        """
        _, (h_n, _) = self.node_history_encoder(node_history)
        encoded = h_n.squeeze(0)
        if self.dropout > 0:
            encoded = F.dropout(encoded, p=self.dropout, training=self.training)
        return encoded

    # =========================================================================
    # ENCODE EDGES
    # =========================================================================
    def encode_edges(
        self,
        ego_history: torch.Tensor,
        neighbor_histories: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode edge interactions between ego and each neighbor.

        Args:
            ego_history:        [batch, H, state_dim]
            neighbor_histories: [batch, num_neighbors, H, state_dim]
        Returns:
            edge_encodings: [batch, num_neighbors, enc_rnn_dim_edge]
        """
        B, N_nbrs, H, D = neighbor_histories.shape

        if N_nbrs == 0:
            return torch.zeros(B, 0, self.enc_rnn_dim_edge,
                               device=ego_history.device)

        ego_expanded = ego_history.unsqueeze(1).expand(B, N_nbrs, H, D)
        joint_flat   = torch.cat([neighbor_histories, ego_expanded], dim=-1) \
                            .reshape(B * N_nbrs, H, 2 * D)

        _, (h_n, _) = self.edge_encoder(joint_flat)
        edge_enc = h_n.squeeze(0)   # [B*N_nbrs, enc_rnn_dim_edge]

        if self.dropout > 0:
            edge_enc = F.dropout(edge_enc, p=self.dropout, training=self.training)

        return edge_enc.reshape(B, N_nbrs, self.enc_rnn_dim_edge)

    # =========================================================================
    # COMBINE EDGE INFLUENCE
    # =========================================================================
    def combine_edge_influence(
        self,
        edge_encodings: torch.Tensor,
        node_history_encoded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate edge encodings into a single influence vector.

        Args:
            edge_encodings:      [batch, num_neighbors, enc_rnn_dim_edge]
            node_history_encoded:[batch, enc_rnn_dim_history]
        Returns:
            combined: [batch, eie_output_dims]
        """
        B = edge_encodings.shape[0]

        if edge_encodings.shape[1] == 0:
            return torch.zeros(B, self.eie_output_dims,
                               device=edge_encodings.device)

        if self.edge_influence_combine == "attention":
            context, _ = self.edge_influence_encoder(edge_encodings,
                                                     node_history_encoded)
            if self.dropout > 0:
                context = F.dropout(context, p=self.dropout, training=self.training)
            return context
        elif self.edge_influence_combine == "mean":
            return edge_encodings.mean(dim=1)
        else:   # "sum" or fallback
            return edge_encodings.sum(dim=1)

    # =========================================================================
    # ENCODE NODE FUTURE
    # =========================================================================
    def encode_node_future(
        self,
        node_present: torch.Tensor,
        node_future: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode agent's ground-truth future trajectory (training only).

        Args:
            node_present: [batch, state_dim]        current state (for h0 init)
            node_future:  [batch, T, pred_state_dim]
        Returns:
            future_enc: [batch, 4 * enc_rnn_dim_future]
        """
        initial_h = torch.stack([
            self.future_encoder_initial_h(node_present),
            torch.zeros_like(self.future_encoder_initial_h(node_present)),
        ], dim=0)

        initial_c = torch.stack([
            self.future_encoder_initial_c(node_present),
            torch.zeros_like(self.future_encoder_initial_c(node_present)),
        ], dim=0)

        _, (h_n, c_n) = self.node_future_encoder(node_future,
                                                  (initial_h, initial_c))

        # h_n / c_n: [2, batch, enc_rnn_dim_future]
        future_enc = torch.cat([h_n[0], h_n[1], c_n[0], c_n[1]], dim=-1)

        if self.dropout > 0:
            future_enc = F.dropout(future_enc, p=self.dropout, training=self.training)

        return future_enc

    # =========================================================================
    # PRIOR  p(z|x)
    # =========================================================================
    def p_z_x(self, x: torch.Tensor, mode: ModeKeys) -> torch.Tensor:
        """Compute prior logits [batch, N, K] from encoder context x."""
        h = self.p_z_x_mlp(x) if self.p_z_x_mlp is not None else x
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.latent.dist_from_h(self.hx_to_z(h), mode)

    # =========================================================================
    # POSTERIOR  q(z|x,y)
    # =========================================================================
    def q_z_xy(self, x: torch.Tensor, y_e: torch.Tensor,
               mode: ModeKeys) -> torch.Tensor:
        """Compute posterior logits [batch, N, K] from context + future."""
        xy = torch.cat([x, y_e], dim=-1)
        h  = self.q_z_xy_mlp(xy) if self.q_z_xy_mlp is not None else xy
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.latent.dist_from_h(self.hxy_to_z(h), mode)

    # =========================================================================
    # FORWARD
    # =========================================================================
    def forward(
        self,
        ego_history: torch.Tensor,
        neighbor_histories: torch.Tensor,
        ego_future: torch.Tensor = None,
        mode: ModeKeys           = ModeKeys.TRAIN,
        num_samples: int         = 1,
    ):
        """
        Full encoder forward pass.

        Args:
            ego_history:        [batch, H, state_dim]
            neighbor_histories: [batch, N_nbrs, H, state_dim]
            ego_future:         [batch, T, pred_state_dim]  (training only)
            mode:               ModeKeys.TRAIN | EVAL | PREDICT
            num_samples:        Number of latent samples

        Returns:
            x:        [batch, x_size]           encoder context
            z:        [num_samples, batch, z_dim] latent samples
            kl_loss:  scalar tensor or None
        """
        # 1. Node history
        node_enc       = self.encode_node_history(ego_history)

        # 2. Edge encodings
        edge_enc       = self.encode_edges(ego_history, neighbor_histories)

        # 3. Edge influence
        edge_influence = self.combine_edge_influence(edge_enc, node_enc)

        # 4. Concatenate encoder context
        x = torch.cat([node_enc, edge_influence], dim=-1)   # [batch, x_size]

        # 5. CVAE: compute distributions and sample
        if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
            assert ego_future is not None, \
                "ego_future required for TRAIN / EVAL mode"

            node_present           = ego_history[:, -1, :]
            y_e                    = self.encode_node_future(node_present, ego_future)
            self.latent.q_dist     = self.q_z_xy(x, y_e, mode)
            self.latent.p_dist     = self.p_z_x(x, mode)
            z                      = self.latent.sample_q(num_samples, mode)
            kl_loss                = self.latent.kl_q_p()

        else:   # PREDICT
            self.latent.p_dist     = self.p_z_x(x, mode)
            z, num_samples, _      = self.latent.sample_p(num_samples, mode)
            kl_loss                = None

        return x, z, kl_loss
