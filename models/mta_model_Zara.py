import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DeepLSTMModel_Zara(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dt=0.1,
                 target_avg_speed=0.0278, speed_tolerance=0.15):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dt = dt

        self.input_embed = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)
        self.postprocess = nn.Linear(input_size, input_size)  # Added postprocess layer

        self.speed_tolerance = speed_tolerance
        self.target_avg_speed = target_avg_speed
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)

    def apply_speed_constraints(self, preds, last_pos):
        B, A, T, D = preds.shape
        constrained_preds = preds.clone()
        current_pos = last_pos.clone()
        for t in range(T):
            displacement = constrained_preds[:, :, t] - current_pos
            speeds = torch.norm(displacement, dim=-1, keepdim=True)
            non_zero_mask = speeds > 0
            clipped_speeds = torch.clamp(speeds, self.min_speed, self.max_speed)
            final_speeds = torch.where(non_zero_mask, clipped_speeds, speeds)
            direction = displacement / (speeds + 1e-8)
            constrained_preds[:, :, t] = current_pos + direction * final_speeds
            current_pos = constrained_preds[:, :, t].clone()
        return constrained_preds

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder_step(self, input_step, hx):
        return self.decoder(input_step, hx)

    def forward(self, history, neighbors=None, goal=None):
        # neighbors and goal kept for compatibility but unused here
        B, A, H, D = history.shape
        device = history.device

        hist_flat = history.reshape(B * A, H, D)
        hist_embedded = self.input_embed(hist_flat)

        # Encoder with gradient checkpointing
        _, (h_n, c_n) = checkpoint(self.forward_encoder, hist_embedded)

        pred_flat = torch.zeros(B * A, 12, D, device=device)

        # Prepare initial hidden/cell states for decoder by repeating last encoder layer's h/c
        h_n = h_n[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_n = c_n[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)
        hx = (h_n, c_n)

        for t in range(12):
            current_pred = pred_flat[:, t:t + 1].clone()
            pred_embedded = self.input_embed(current_pred)
            out, hx = checkpoint(self.forward_decoder_step, pred_embedded, hx)
            step_output = self.output(out.squeeze(1))
            step_output = self.postprocess(step_output)  # Postprocess here
            pred_flat[:, t] = step_output

        predictions = pred_flat.view(B, A, 12, D)
        last_known_pos = history[:, :, -1, :]
        constrained_preds = self.apply_speed_constraints(predictions, last_known_pos)

        return constrained_preds, None, None  # No coeff mean/var needed