import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# Potential Field
class PotentialField(nn.Module):
    def __init__(self, goal, num_agents=64, k_init=1.0, repulsion_radius=0.5):
        super().__init__()
        self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))
        self.repulsion_radius = repulsion_radius

        # 3 coefficients are to be learned
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)

    def forward(self, pos, predicted, neighbors, goal, coeffs): #RK using input coeffts and goal
        """
        Compute the APF-based force vector for each agent.

        Parameters:
        - pos:       [B, A, 2]      - current agent positions
        - predicted: [B, A, 1, 2]   - predicted next positions (single step)
        - neighbors: [B, A, N, 2]   - neighboring agents' current positions
        - goal:      [B, A, 2]      - true goal positions (final ground-truth)
        - coeffs:    [B, A, 3]      - APF coefficients per agent (k_att1, k_att2, k_rep)

        Returns:
        - total_force: [B, A, 2]
        - coeffs:      [B, A, 3]
        """

        # Split coefficients
        k_att1 = coeffs[..., 0:1]  # for attractive force to goal
        k_att2 = coeffs[..., 1:2]  # for attractive force to predicted
        k_rep  = coeffs[..., 2:3]  # for repulsive force

        # --- Attractive Forces ---
        F_goal = k_att1 * (goal - pos)                    # [B, A, 2]
        F_pred = k_att2 * (predicted[:, :, 0, :] - pos)   # [B, A, 2]

        # --- Repulsive Forces ---
        diffs = pos.unsqueeze(2) - neighbors              # [B, A, N, 2]
        dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6  # [B, A, N, 1]
        mask = (dists < self.repulsion_radius).float()    # Only repel nearby neighbors

        F_rep = (k_rep.unsqueeze(2) * diffs / dists.pow(2) * mask).sum(dim=2)  # [B, A, 2]

        # --- Total Force ---
        total_force = F_goal + F_pred + F_rep

        return total_force, coeffs



class IntegratedMTAPFMModel_Zara(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, goal=(4.2, 4.2)):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_embed = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.postprocess = nn.Linear(input_size, input_size)  # RK: extra dense layer after decoder output
        self.output = nn.Linear(hidden_size, input_size)
        self.pfm = PotentialField(goal=goal)

        # Speed constraint parameters
        self.target_avg_speed = 0.0278 # From your calculated average
        self.speed_tolerance = 0.15     # 15% tolerance
        self.min_speed = self.target_avg_speed * (1 - self.speed_tolerance)  # 0.2547
        self.max_speed = self.target_avg_speed * (1 + self.speed_tolerance)  # 0.3445

    def apply_speed_constraints(self, predictions, last_positions):
        """
        Apply speed constraints to predictions by clipping speeds to thresholds
        Args:
            predictions: [B, A, T, 2] predicted trajectories
            last_positions: [B, A, 2] last known positions (from history)
        Returns:
            constrained_predictions: [B, A, T, 2] speed-constrained predictions
        """
        B, A, T, D = predictions.shape
        constrained_preds = predictions.clone()

        # Start from the last known position
        current_pos = last_positions.clone()  # [B, A, 2]

        for t in range(T):
            # Calculate displacement from current position to predicted next position
            displacement = predictions[:, :, t] - current_pos  # [B, A, 2]

            # Calculate speed (distance) for this step
            speeds = torch.norm(displacement, dim=-1, keepdim=True)  # [B, A, 1]

            # Clip speeds to the allowed range
            # Don't clip zero speeds (padding), only clip actual movements
            non_zero_mask = speeds > 0
            clipped_speeds = torch.clamp(speeds, self.min_speed, self.max_speed)

            # Only apply clipping to non-zero speeds (preserve padding)
            final_speeds = torch.where(non_zero_mask, clipped_speeds, speeds)

            # Calculate new displacement with clipped speed
            # Keep the direction, but adjust the magnitude
            direction = displacement / (speeds + 1e-8)  # Unit direction vector
            clipped_displacement = direction * final_speeds

            # Update constrained prediction for this timestep
            constrained_preds[:, :, t] = current_pos + clipped_displacement

            # Update current position for next iteration
            current_pos = constrained_preds[:, :, t].clone()

        return constrained_preds

    def forward(self, history, neighbors, goal):  # RK: add goal as input here
        B, A, H, D = history.shape
        device = history.device

        agent_ids = torch.arange(A).repeat(B, 1).to(device)  # [B, A]

        # LSTM encoding
        hist_flat = history.reshape(B * A, H, D)
        hist_embedded = self.input_embed(hist_flat)
        _, (h_n, c_n) = self.encoder(hist_embedded)

        # LSTM decoding
        pred_flat = torch.zeros(B * A, 12, D, device=device)
        h_n = h_n[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)

        for t in range(12):
            current_pred = pred_flat[:, t:t+1].clone()
            pred_embedded = self.input_embed(current_pred)
            out, (h_n, c_n) = self.decoder(pred_embedded, (h_n, c_n))
            step_output = self.output(out.squeeze(1))            # [B*A, D]
            pred_flat[:, t] = self.postprocess(step_output)       # RK: apply additional dense layer here

        predictions = pred_flat.view(B, A, 12, D)

        # PFM adjustment
        adjusted_preds = torch.zeros_like(predictions)
        current_pos = history[:, :, -1, :].clone()  # Initial position
        coeff_list = []

        coeffs = self.pfm.coeff_embedding(agent_ids)  # RK: get APF coefficients from embedding

        for t in range(12):
            pred_slice = predictions[:, :, t:t+1].clone()
            forces, coeff_step = self.pfm(  # RK: pass APF coefficients and goal
                current_pos, pred_slice, neighbors.clone(), goal, coeffs
            )

            if t == 0:
                adjusted_preds[:, :, t] = current_pos + forces
            else:
                adjusted_preds[:, :, t] = adjusted_preds[:, :, t - 1] + forces

            coeff_list.append(coeff_step)

            current_pos = adjusted_preds[:, :, t].clone()  # RK: update current_pos here

        # Apply speed constraints after PFM adjustments
        last_known_pos = history[:, :, -1, :]
        constrained_preds = self.apply_speed_constraints(adjusted_preds, last_known_pos)

        coeff_stack = torch.stack(coeff_list, dim=0)  # [12, B, A, 3]
        coeff_mean = coeff_stack.mean()
        coeff_var = coeff_stack.var(unbiased=False)

        return constrained_preds, coeff_mean, coeff_var