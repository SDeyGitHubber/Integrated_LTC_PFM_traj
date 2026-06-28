import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class PotentialField(nn.Module):
    def __init__(self, goal, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
        super().__init__()
        self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))
        self.repulsion_radius = repulsion_radius
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)

    def forward(self, pos, predicted, neighbors, goal, coeffs):
        k_att1 = coeffs[..., 0:1]
        k_att2 = coeffs[..., 1:2]
        k_rep = coeffs[..., 2:3]

        F_goal = k_att1 * (goal - pos)
        F_pred = k_att2 * (predicted[:, :, 0, :] - pos)

        diffs = pos.unsqueeze(2) - neighbors
        dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
        mask = (dists < self.repulsion_radius).float()

        F_rep = (k_rep.unsqueeze(2) * diffs / dists.pow(2) * mask).sum(dim=2)

        total_force = F_goal + F_pred + F_rep

        return total_force, coeffs


class CheckpointedIntegratedMTAPFMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2,
                 goal=(4.2,4.2), target_avg_speed=4.087,
                 speed_tolerance=0.15, num_agents=1000):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.input_embed = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # If postprocess is unnecessary, replace with identity
        # self.postprocess = nn.Identity()
        self.postprocess = nn.Linear(input_size, input_size)
        self.coeff_projector = nn.Linear(hidden_size, 3) # RKL hidden_dim is the dimensionality of the LSTM’s output per agent.

        self.output = nn.Linear(hidden_size, input_size)
        self.pfm = PotentialField(goal=goal, num_agents=num_agents)

        if target_avg_speed is None:
            raise ValueError("target_avg_speed must be computed from dataset and passed here.")

        self.target_avg_speed = target_avg_speed
        self.speed_tolerance = speed_tolerance
        self.min_speed = self.target_avg_speed * (1 - self.speed_tolerance)
        self.max_speed = self.target_avg_speed * (1 + self.speed_tolerance)

    def apply_speed_constraints(self, predictions, last_positions):
        B, A, T, D = predictions.shape
        constrained_preds = predictions.clone()
        current_pos = last_positions.clone()
        for t in range(T):
            displacement = predictions[:, :, t] - current_pos
            speeds = torch.norm(displacement, dim=-1, keepdim=True)
            non_zero_mask = speeds > 0
            clipped_speeds = torch.clamp(speeds, self.min_speed, self.max_speed)
            final_speeds = torch.where(non_zero_mask, clipped_speeds, speeds)
            direction = displacement / (speeds + 1e-8)
            clipped_displacement = direction * final_speeds
            constrained_preds[:, :, t] = current_pos + clipped_displacement
            current_pos = constrained_preds[:, :, t].clone()
        return constrained_preds

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder_step(self, input_step, hx):
        return self.decoder(input_step, hx)

    def forward(self, history, neighbors, goal):
        B, A, H, D = history.shape
        device = history.device
        agent_ids = torch.arange(A).repeat(B, 1).to(device)

        hist_flat = history.reshape(B * A, H, D)
        hist_embedded = self.input_embed(hist_flat)

        # Gradient checkpoint on encoder
        _, (h_n, c_n) = checkpoint.checkpoint(self.forward_encoder, hist_embedded)

        pred_flat = torch.zeros(B * A, 12, D, device=device)

        h_n = h_n[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_n = c_n[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)
        hx = (h_n, c_n)

        for t in range(12):
            current_pred = pred_flat[:, t:t+1].clone()
            pred_embedded = self.input_embed(current_pred)
            out, hx = checkpoint.checkpoint(self.forward_decoder_step, pred_embedded, hx)
            step_output = self.output(out.squeeze(1))
            pred_flat[:, t] = self.postprocess(step_output)

        predictions = pred_flat.view(B, A, 12, D)
        adjusted_preds = torch.zeros_like(predictions)
        current_pos = history[:, :, -1, :].clone()
        coeff_list = []

        coeffs = self.pfm.coeff_embedding(agent_ids)
        print("hx shape (h_n[-1]):", h_n[-1].shape)
        coeffs = self.coeff_projector(h_n[-1])  # Selecting the last layer’s hidden state for each batch element. # project hx (last hidden/cell state) into coeff
        coeffs = coeffs.view(B, A, 3)
        print("coeffs shape after projection:", coeffs.shape) 

        for t in range(12):
            pred_slice = predictions[:, :, t:t+1].clone()
            forces, coeff_step = self.pfm(current_pos, pred_slice, neighbors.clone(), goal, coeffs) #Then pass this reshaped coeffs in your call to pfm:
            if t == 0:
                adjusted_preds[:, :, t] = current_pos + forces
            else:
                adjusted_preds[:, :, t] = adjusted_preds[:, :, t-1] + forces
            coeff_list.append(coeff_step)
            current_pos = adjusted_preds[:, :, t].clone()

        last_known_pos = history[:, :, -1, :]
        constrained_preds = self.apply_speed_constraints(adjusted_preds, last_known_pos)

        coeff_stack = torch.stack(coeff_list, dim=0)
        coeff_mean = coeff_stack.mean()
        coeff_var = coeff_stack.var(unbiased=False)

        return constrained_preds, coeff_mean, coeff_var