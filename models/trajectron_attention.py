"""
================================================================================
TRAJECTRON++ — ADDITIVE ATTENTION
================================================================================

Bahdanau-style additive attention used by TrajectronEncoder to combine
edge-influence encodings across multiple neighbors.

    score(s_i, h_j) = v^T tanh(W_1 s_i + W_2 h_j)

Used by:
  TrajectronEncoder  (models/trajectron_encoder.py)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.

    Combines a set of encoder hidden states (edge encodings) by attending over
    them using a query (node history encoding).

    Args:
        encoder_dim (int): Dimension of encoder (edge) hidden states
        decoder_dim (int): Dimension of decoder (node history) hidden state
    """

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.W1 = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.W2 = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.v  = nn.Linear(decoder_dim, 1,            bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ):
        """
        Args:
            encoder_outputs: [batch, num_edges, encoder_dim]
            decoder_hidden:  [batch, decoder_dim]
        Returns:
            context:           [batch, encoder_dim]   weighted sum of edge encodings
            attention_weights: [batch, num_edges]     softmax scores
        """
        query  = self.W1(decoder_hidden).unsqueeze(1)   # [B, 1, decoder_dim]
        keys   = self.W2(encoder_outputs)                 # [B, num_edges, decoder_dim]
        energy = self.v(torch.tanh(query + keys)).squeeze(-1)  # [B, num_edges]

        attention_weights = F.softmax(energy, dim=-1)
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)                                     # [B, encoder_dim]

        return context, attention_weights
