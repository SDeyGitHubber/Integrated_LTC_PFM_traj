# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint


# class PotentialField(nn.Module):
#     def __init__(self, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
#         super().__init__()
#         self.repulsion_radius = repulsion_radius
#         self.coeff_embedding = nn.Embedding(num_agents, 3)
#         self.coeff_embedding.weight.data.fill_(k_init)

#     def forward(self, pos, predicted, neighbors, goal, coeffs):
#         print("[PFM] pos.shape:", pos.shape)
#         print("[PFM] predicted.shape:", predicted.shape)
#         print("[PFM] neighbors.shape:", neighbors.shape)
#         print("[PFM] goal.shape:", goal.shape)
#         print("[PFM] coeffs.shape:", coeffs.shape)

#         k_att1 = coeffs[..., 0:1]
#         k_att2 = coeffs[..., 1:2]
#         k_rep = coeffs[..., 2:3]

#         F_goal = k_att1 * (goal - pos)

#         # Handle predicted dim cases for ego or neighbors
#         if predicted.dim() == 3:
#             F_pred = k_att2 * (predicted - pos)
#         else:
#             F_pred = k_att2 * (predicted[:, :, 0, :] - pos)

#         # ROBUST repulsion handling - completely avoid computation when no neighbors
#         if neighbors.size(1) == 0:
#             F_rep = torch.zeros_like(pos)
#             print("[PFM] No neighbors - F_rep set to zero with shape:", F_rep.shape)
#         else:
#             # Only compute when neighbors actually exist
#             diffs = pos.unsqueeze(2) - neighbors  # [N, 1, 2] - [N, M, 2] = [N, M, 2]
#             dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6  # [N, M, 1]
#             mask = (dists < self.repulsion_radius).float()  # [N, M, 1] #RKL: Add and dists>0.00001
            
#             # Ensure k_rep broadcasting matches diffs dimensions exactly
#             k_rep_expanded = k_rep.unsqueeze(2)  # [N, 1, 1]
#             repulsion_per_neighbor = k_rep_expanded * diffs / dists.pow(2) * mask  # [N, M, 2] #RKL: max(dists, 0.000001)
#             F_rep = repulsion_per_neighbor.sum(dim=2)  # [N, 2]
            
#             print("[PFM] Computed F_rep with shape:", F_rep.shape)

#         total_force = F_goal + F_pred + F_rep
#         print("[PFM] Total force shape:", total_force.shape)

#         return total_force, coeffs


# class CheckpointedIntegratedMTAPFMModel_neighbours(nn.Module):
#     def __init__(self, input_size=2, hidden_size=64, num_layers=2,
#                  target_avg_speed=4.087, speed_tolerance=0.15, num_agents=1000):
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size

#         self.input_embed = nn.Linear(input_size, hidden_size)
#         self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

#         self.postprocess = nn.Linear(input_size, input_size)
#         self.coeff_projector_ego = nn.Linear(hidden_size, 3)
#         self.coeff_projector_neighbors = nn.Linear(hidden_size, 3)

#         self.output = nn.Linear(hidden_size, input_size)
#         self.pfm = PotentialField(num_agents=num_agents)

#         if target_avg_speed is None:
#             raise ValueError("target_avg_speed must be computed from dataset and passed here.")

#         self.target_avg_speed = target_avg_speed
#         self.speed_tolerance = speed_tolerance
#         self.min_speed = self.target_avg_speed * (1 - self.speed_tolerance)
#         self.max_speed = self.target_avg_speed * (1 + self.speed_tolerance)
#         self.dt = 0.1  # timestep used for potential field motion update

#     def apply_speed_constraints(self, predictions, last_positions):
#         B, A, T, D = predictions.shape
#         constrained_preds = predictions.clone()
#         current_pos = last_positions.clone()
#         for t in range(T):
#             displacement = predictions[:, :, t] - current_pos
#             speeds = torch.norm(displacement, dim=-1, keepdim=True)
#             non_zero_mask = speeds > 0
#             clipped_speeds = torch.clamp(speeds, self.min_speed, self.max_speed)
#             final_speeds = torch.where(non_zero_mask, clipped_speeds, speeds)
#             direction = displacement / (speeds + 1e-8)
#             clipped_displacement = direction * final_speeds
#             constrained_preds[:, :, t] = current_pos + clipped_displacement
#             current_pos = constrained_preds[:, :, t].clone()
#         return constrained_preds

#     def forward_encoder(self, x):
#         return self.encoder(x)

#     def forward_decoder_step(self, input_step, hx):
#         return self.decoder(input_step, hx)

#     def forward(self, history_neighbors, goal): #RKL: in data loader, while copying data to histor_neighbours also copy their goal, such that shape of  history_neighbours and goal match
#         """
#         history_neighbors: [B, A, max_neighbors+1, H, 2]
#            neighbors dim 0 = ego, others = neighbors
#         goal: [B, A, 2]
#         """
#         print("[MODEL] forward: history_neighbors.shape =", history_neighbors.shape)
#         print("[MODEL] forward: goal.shape =", goal.shape)

#         B, A, max_nb_plus_1, H, D = history_neighbors.shape
#         device = history_neighbors.device

#         histories_flat = history_neighbors.reshape(B * A * max_nb_plus_1, H, D)
#         print("[MODEL] histories_flat.shape after reshape:", histories_flat.shape)

#         embedded = self.input_embed(histories_flat)
#         print("[MODEL] embedded.shape after input_embed:", embedded.shape)

#         _, (h_n, c_n) = checkpoint.checkpoint(self.forward_encoder, embedded)

#         print("[MODEL] h_n.shape:", h_n.shape)
#         print("[MODEL] c_n.shape:", c_n.shape)

#         h_top = h_n[-1].view(B, A, max_nb_plus_1, self.hidden_size)
#         print("[MODEL] h_top.shape after reshape:", h_top.shape)

#         h_ego = h_top[:, :, 0, :] #RKL: Delete
#         h_neighbors = h_top[:, :, 1:, :] #RKL: Delete

#         print("[MODEL] h_ego.shape:", h_ego.shape) #RKL: Delete
#         print("[MODEL] h_neighbors.shape:", h_neighbors.shape) #RKL: Delete

#         coeffs_ego = self.coeff_projector_ego(h_ego) #RKL: On complete h reshaped to B*A*13, H=8, D=2
#         coeffs_neighbors = self.coeff_projector_neighbors(h_neighbors) #RKL: Deleted

#         print("[MODEL] coeffs_ego.shape:", coeffs_ego.shape) #RKL: replaced by coeefs
#         print("[MODEL] coeffs_neighbors.shape:", coeffs_neighbors.shape) #RKL: Deleted

#         pred_flat = torch.zeros(B * A, 12, D, device=device) #RKL: change B*A to B*A*(neigh+1), dimensions B*A*13, 12, D
#         h_n_decoder = h_n[:, :B * A, :].contiguous() #RKL: B*A to above??
#         c_n_decoder = c_n[:, :B * A, :].contiguous() #RKL: B*A to above??
#         hx = (h_n_decoder, c_n_decoder)

#         for t in range(12):
#             current_pred = pred_flat[:, t:t + 1].clone()
#             print(f"[MODEL] Step {t} - current_pred.shape:", current_pred.shape)

#             pred_embedded = self.input_embed(current_pred)
#             print(f"[MODEL] Step {t} - pred_embedded.shape:", pred_embedded.shape)

#             h, c = hx
#             hx = (h.contiguous(), c.contiguous())
#             print(f"[MODEL] Step {t} - hx shapes: h {h.shape}, c {c.shape}")

#             out, hx = checkpoint.checkpoint(self.forward_decoder_step, pred_embedded, hx)
#             print(f"[MODEL] Step {t} - decoder out.shape:", out.shape)

#             step_output = self.output(out.squeeze(1))
#             print(f"[MODEL] Step {t} - step_output.shape:", step_output.shape)

#             pred_flat[:, t] = self.postprocess(step_output)
#             print(f"[MODEL] Step {t} - pred_flat state updated")

#         predictions = pred_flat.view(B, A, 12, D) #RKL: add dimension of (neigh+1) B, A, neigh+1=13, 12, D
#         print("[MODEL] predictions.shape after view:", predictions.shape)
#         adjusted_preds = torch.zeros_like(predictions)
#         print("[MODEL] initialized adjusted_preds.shape:", adjusted_preds.shape)

#         current_pos_ego = history_neighbors[:, :, 0, -1, :].clone() #RKL: current_pos = history_neighbors[:, :, :, -1, :].clone
#         current_pos_neighbors = history_neighbors[:, :, 1:, -1, :].clone() #RKL: Delete

#         print("[MODEL] current_pos_ego.shape:", current_pos_ego.shape) #RKL: Delete
#         print("[MODEL] current_pos_neighbors.shape:", current_pos_neighbors.shape) #RKL: Delete

#         coeff_list = []

#         for t in range(12): #add loop on a= 0 to neigh+1
#             pred_slice_ego = predictions[:, :, t:t + 1].clone() #RKL: predictions[:, :, a, t+1]
#             print(f"[MODEL] neighbor loop step {t} - pred_slice_ego.shape:", pred_slice_ego.shape)

#             forces_ego, coeff_step_ego = self.pfm(current_pos_ego, pred_slice_ego,
#                                                  current_pos_neighbors, goal, coeffs_ego) 
#             #RKL: current_pos_ego -> current_pos[agent a in its dimension] -- match shape
#             #RKL: pred_slice_ego ->pred_slice_ego[agent a in its dimension] -- match shape
#             #RKL: goal -> goal[agent a in its dimension]
#             #RKL: coeffs_ego -> coeffs[agent a in its dimension]
#             # current_pos_neighbors -> curret_pos (all agents including ego)

#             print(f"[MODEL] neighbor loop step {t} - forces_ego.shape:", forces_ego.shape)
#             print(f"[MODEL] neighbor loop step {t} - coeff_step_ego.shape:", coeff_step_ego.shape)

#             if t == 0:
#                 adjusted_preds[:, :, t] = current_pos_ego + forces_ego #RKL: adjusted_preds[:, :, a, t]
#             else:
#                 adjusted_preds[:, :, t] = adjusted_preds[:, :, t - 1] + forces_ego #RKL: adjusted_preds[:, :, a, t] and #RKL: adjusted_preds[:, :, a, t-1]

#             current_pos_ego = adjusted_preds[:, :, t].clone() #RKL: current_pos[a] = adjusted_preds[:, :, a, t]
#             print(f"[MODEL] neighbor loop step {t} - updated current_pos_ego.shape:", current_pos_ego.shape)

#             #RKL: Delete from here
#             B_, A_, N_ = current_pos_neighbors.shape[:3]
#             cur_neighbors_flat = current_pos_neighbors.reshape(B_ * A_ * N_, 2)
#             pred_slice_neighbors = cur_neighbors_flat.unsqueeze(1)
#             coeffs_neighbors_flat = coeffs_neighbors.reshape(B_ * A_ * N_, 3)

#             print(f"[MODEL] neighbor loop step {t} - current_pos_neighbors.shape:", current_pos_neighbors.shape)
#             print(f"[MODEL] neighbor loop step {t} - cur_neighbors_flat.shape:", cur_neighbors_flat.shape)
#             print(f"[MODEL] neighbor loop step {t} - pred_slice_neighbors.shape:", pred_slice_neighbors.shape)
#             print(f"[MODEL] neighbor loop step {t} - coeffs_neighbors_flat.shape:", coeffs_neighbors_flat.shape)

#             neighbors_neighbors = torch.empty(cur_neighbors_flat.shape[0], 0, 2, device=device) #RKL - incorrect
#             goal_expanded_neighbors = goal.unsqueeze(2).expand(B_, A_, N_, 2).contiguous().reshape(B_ * A_ * N_, 2) #RKL - incorrect??

#             print(f"[MODEL] neighbor loop step {t} - neighbors_neighbors.shape:", neighbors_neighbors.shape) 
#             print(f"[MODEL] neighbor loop step {t} - goal_expanded_neighbors.shape:", goal_expanded_neighbors.shape)

#             forces_neighbors, _ = self.pfm(
#                 pos=cur_neighbors_flat,
#                 predicted=pred_slice_neighbors,
#                 neighbors=neighbors_neighbors,
#                 goal=goal_expanded_neighbors,
#                 coeffs=coeffs_neighbors_flat
#             )
            
#             print(f"[MODEL] neighbor loop step {t} - forces_neighbors.shape after PFM:", forces_neighbors.shape)
#             print(f"[MODEL] neighbor loop step {t} - cur_neighbors_flat.numel():", cur_neighbors_flat.numel())
#             print(f"[MODEL] neighbor loop step {t} - forces_neighbors.numel():", forces_neighbors.numel())

#             # Force correct shape if PFM returns unexpected dimensions
#             expected_shape = cur_neighbors_flat.shape  # [N, 2]
#             if forces_neighbors.shape != expected_shape:
#                 print(f"[FIX] forces_neighbors shape mismatch! Expected: {expected_shape}, Got: {forces_neighbors.shape}")
#                 forces_neighbors = torch.zeros_like(cur_neighbors_flat)
#                 print(f"[FIX] Forced forces_neighbors to correct shape: {forces_neighbors.shape}")

#             nextp_neighbors = cur_neighbors_flat + forces_neighbors * self.dt

#             print(f"[MODEL] neighbor loop step {t} - nextp_neighbors.shape before view:", nextp_neighbors.shape)
#             print(f"[MODEL] neighbor loop step {t} - expected shape for view: ({B_}, {A_}, {N_}, 2)")

#             try:
#                 current_pos_neighbors = nextp_neighbors.view(B_, A_, N_, 2).clone().detach()
#                 print(f"[MODEL] neighbor loop step {t} - successfully reshaped current_pos_neighbors")
#             except Exception as e:
#                 print(f"[ERROR] neighbor loop step {t} - Failed to view nextp_neighbors:", e)
#                 raise

#             coeff_list.append(coeff_step_ego)

#         last_known_pos_ego = history_neighbors[:, :, 0, -1, :]
#         constrained_preds = self.apply_speed_constraints(adjusted_preds, last_known_pos_ego) #RKL: this should be inside above loop without additional time loop

#         coeff_stack = torch.stack(coeff_list, dim=0)
#         coeff_mean = coeff_stack.mean()
#         coeff_var = coeff_stack.var(unbiased=False)

#         print("[MODEL] forward completed: returning predictions and coeff stats")

#         return constrained_preds, coeff_mean, coeff_var


# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint

# class PotentialField(nn.Module):
#     def __init__(self, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
#         super().__init__()
#         self.repulsion_radius = repulsion_radius
#         self.coeff_embedding = nn.Embedding(num_agents, 3)
#         self.coeff_embedding.weight.data.fill_(k_init)
    
#     def forward(self, pos, predicted, neighbors, goal, coeffs):
#         print("[PFM] pos.shape:", pos.shape)
#         print("[PFM] predicted.shape:", predicted.shape)
#         print("[PFM] neighbors.shape:", neighbors.shape)
#         print("[PFM] goal.shape:", goal.shape)
#         print("[PFM] coeffs.shape:", coeffs.shape)
        
#         k_att1 = coeffs[..., 0:1]
#         k_att2 = coeffs[..., 1:2]
#         k_rep = coeffs[..., 2:3]
        
#         F_goal = k_att1 * (goal - pos)
        
#         # Handle predicted dim cases for ego or neighbors
#         if predicted.dim() == 3:
#             F_pred = k_att2 * (predicted - pos)
#         else:
#             F_pred = k_att2 * (predicted[:, :, 0, :] - pos)
        
#         # ROBUST repulsion handling - completely avoid computation when no neighbors
#         if neighbors.size(1) == 0:
#             F_rep = torch.zeros_like(pos)
#             print("[PFM] No neighbors - F_rep set to zero with shape:", F_rep.shape)
#         else:
#             diffs = pos.unsqueeze(2) - neighbors  # [N, 1, 2] - [N, M, 2] = [N, M, 2]
#             dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
            
#             mask = (dists < self.repulsion_radius) & (dists > 0.00001)  #RKL mask
#             mask = mask.float()
#             print("[PFM] Applied enhanced mask")
            
#             k_rep_expanded = k_rep.unsqueeze(2)
#             safe_dists = torch.max(dists, torch.tensor(0.000001, device=dists.device))
#             repulsion = k_rep_expanded * diffs / safe_dists.pow(2) * mask
#             F_rep = repulsion.sum(dim=2)
            
#             print("[PFM] Computed F_rep with shape:", F_rep.shape)
        
#         total_force = F_goal + F_pred + F_rep
#         print("[PFM] Total force shape:", total_force.shape)
        
#         return total_force, coeffs


# class CheckpointedIntegratedMTAPFM_neighbours(nn.Module):
#     def __init__(self, input_size=2, hidden_size=64, num_layers=2,
#                  target_avg_speed=4.087, speed_tolerance=0.15, num_agents=1000):
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
        
#         self.input_embed = nn.Linear(input_size, hidden_size)
#         self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
#         self.postprocess = nn.Linear(input_size, input_size)
#         self.coeff_projector = nn.Linear(hidden_size, 3)  # Single projector for all entities
#         print("[INIT] Using unified coefficient projector for all entities")
        
#         self.output = nn.Linear(hidden_size, input_size)
#         self.pfm = PotentialField(num_agents)
        
#         if target_avg_speed is None:
#             raise ValueError("target_avg_speed must be provided")
        
#         self.target_avg_speed = target_avg_speed
#         self.speed_tolerance = speed_tolerance
#         self.min_speed = self.target_avg_speed * (1 - self.speed_tolerance)
#         self.max_speed = self.target_avg_speed * (1 + self.speed_tolerance)
#         self.dt = 0.1  # timestep used for update
    
#     def apply_speed_constraints(self, predictions, last_positions):
#         B, A, entities, T, D = predictions.shape
#         constrained = predictions.clone()
#         current_pos = last_positions.clone()
#         for t in range(T):
#             displacement = constrained[:, :, :, t] - current_pos
#             speeds = torch.norm(displacement, dim=-1, keepdim=True)
#             mask = speeds > 0
#             clipped = torch.clamp(speeds, self.min_speed, self.max_speed)
#             final = torch.where(mask, clipped, speeds)
#             direction = displacement / (speeds + 1e-8)
#             clipped_disp = direction * final
#             constrained[:, :, :, t] = current_pos + clipped_disp
#             current_pos = constrained[:, :, :, t].clone()
#         return constrained
    
#     def forward_encoder(self, x):
#         return self.encoder(x)
    
#     def forward_decoder_step(self, x, hx):
#         return self.decoder(x, hx)
    
#     def forward(self, history_neighbors, goal):  # RKL in comment
#         """
#         history_neighbors: [B, A, max_neighbors+1, H, 2]
#         goal: [B, A, max_neighbors+1, 2]
#         """
#         print("[MODEL] === Forward pass ===")
#         print("[MODEL] history_neighbors shape:", history_neighbors.shape)
#         print("[MODEL] goal shape:", goal.shape)
        
#         B, A, max_nb_plus_1, H, D = history_neighbors.shape
#         device = history_neighbors.device
        
#         assert goal.shape == (B, A, max_nb_plus_1, D), "Goal shape mismatch"  # RKL check
#         print("[MODEL] Goal shape verified")
        
#         histories_flat = history_neighbors.view(B * A * max_nb_plus_1, H, D)
#         embedded = self.input_embed(histories_flat)
        
#         _, (h_n, c_n) = checkpoint.checkpoint(self.forward_encoder, embedded)
        
#         h_out = h_n[-1].view(B, A, max_nb_plus_1, self.hidden_size)
#         coeffs = self.coeff_projector(h_out)  # RKL one projector
        
#         pred_len = 12
#         pred_flat = torch.zeros(B * A * max_nb_plus_1, pred_len, D, device=device)
        
#         hx = (h_n, c_n)
#         last_pos = history_neighbors[:, :, :, -1, :].reshape(B * A * max_nb_plus_1, 1, D)  # RKL prev pos
        
#         for t in range(pred_len):
#             if t == 0:
#                 decoder_in = last_pos.clone()  # RKL1 at t=0
#             else:
#                 decoder_in = pred_flat[:, t-1:t].clone()
#             embedded_in = self.input_embed(decoder_in)
#             h, c = hx
#             hx = (h.contiguous(), c.contiguous())
#             out, hx = checkpoint.checkpoint(self.forward_decoder_step, embedded_in, hx)
#             step_out = self.output(out.squeeze(1))
#             pred_flat[:, t] = self.postprocess(step_out) if t == 0 else pred_flat[:, t-1] + self.postprocess(step_out)  # RKL1 recursive
            
#         decoded_preds = pred_flat.view(B, A, max_nb_plus_1, pred_len, D)  # RKL shape
        
#         # Motion integration with PFM
#         adjusted_preds = torch.zeros_like(decoded_preds)
#         current_pos = history_neighbors[:, :, :, -1, :].clone()
#         coeff_list = []
        
#         for t in range(pred_len):
#             print(f"[MODEL] Time step {t}")
#             for idx in range(max_nb_plus_1):
#                 pos = current_pos[:, :, idx]
#                 pred = decoded_preds[:, :, idx, t:t+1]
#                 goal_vec = goal[:, :, idx]
#                 coeff = coeffs[:, :, idx]
                
#                 neighbors = torch.stack([current_pos[:, :, i] for i in range(max_nb_plus_1) if i != idx], dim=2) \
#                     if max_nb_plus_1 > 1 else torch.empty(B, A, 0, 2, device=device)
                
#                 force, updated_coeff = self.pfm(pos, pred, neighbors, goal_vec, coeff)
                
#                 if t == 0:
#                     new_pos = pos + force
#                 else:
#                     new_pos = adjusted_preds[:, :, idx, t-1] + force
                
#                 if t > 0:  # RKL apply speed constraints
#                     prev_pos = adjusted_preds[:, :, idx, t-1]
#                     disp = new_pos - prev_pos
#                     speed = disp.norm(dim=-1, keepdim=True)
#                     mask = speed > 0
#                     clipped_speed = torch.clamp(speed, self.min_speed, self.max_speed)
#                     disp = disp / (speed + 1e-8) * clipped_speed
#                     new_pos = prev_pos + disp
                
#                 adjusted_preds[:, :, idx, t] = new_pos
                
#                 if idx == 0:
#                     coeff_list.append(updated_coeff)
                    
#             current_pos = adjusted_preds[:, :, :, t].clone()
        
#         coeff_stack = torch.stack(coeff_list)
#         coeff_mean = coeff_stack.mean()
#         coeff_var = coeff_stack.var(unbiased=False)
        
#         print("[MODEL] Forward pass complete. Returning results.")
#         return adjusted_preds, decoded_preds, coeff_mean, coeff_var


import torch.nn as nn
import torch.utils.checkpoint as cp
import torch


# --- Potential Field Module ---
class PotentialField(nn.Module):
    """
    Physics-inspired module that computes social forces on each agent entity in a multi-agent scene.

    This module models attraction to goals, repulsion from neighbors, and attraction to an LSTM-generated prediction,
    following principles of potential field methods for collision-free and goal-directed path planning.

    Args:
        num_agents (int): Number of unique agents (for force coefficient embedding).
        k_init (float): Initial value for all force coefficients.
        repulsion_radius (float): Maximum distance within which repulsion from neighbors is applied.
    """

    def __init__(self, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
        super().__init__()
        self.repulsion_radius = repulsion_radius
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)

    def forward(self, pos, predicted, neighbors, goal, coeffs):
        """
        Computes the total potential field force for each agent/entity at the current timestep.

        Args:
            pos (Tensor): Current positions of agents, shape [B, A, D].
            predicted (Tensor): Raw predicted displacements or positions, shape [B, A, D] or [B, A, 1, D].
            neighbors (Tensor): Neighbor positions, shape [B, A, N, D].
            goal (Tensor): Goal positions for each agent/entity, shape [B, A, D].
            coeffs (Tensor): Force coefficients for attraction/repulsion, shape [B, A, 3].

        Returns:
            total_force (Tensor): Net force (goal attraction + prediction attraction + neighbor repulsion), shape [B, A, D].
            coeffs (Tensor): Same as input, returned for downstream tracking/logging.
        """
        k1 = coeffs[..., 0:1]
        k2 = coeffs[..., 1:2]
        kr = coeffs[..., 2:3]

        F_goal = k1 * (goal - pos)

        if predicted.dim() == 3:
            F_pred = k2 * (predicted - pos)
        else:
            F_pred = k2 * (predicted[:, :, 0, :] - pos)

        if neighbors.size(2) == 0:
            F_rep = torch.zeros_like(pos)
            print("[PFM] No neighbors - F_rep set to zero, shape:", F_rep.shape)
        else:
            diffs = pos.unsqueeze(2) - neighbors
            dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
            mask = (dists < self.repulsion_radius) & (dists > 1e-5)
            mask = mask.float()
            print("[PFM] Applied enhanced mask")

            kr_exp = kr.unsqueeze(2)
            safe_dists = torch.max(dists, torch.tensor(1e-6, device=dists.device))

            repulsion = kr_exp * diffs / safe_dists.pow(2) * mask
            F_rep = repulsion.sum(dim=2)

            print("[PFM] Computed F_rep shape:", F_rep.shape)

        total_force = F_goal + F_pred + F_rep
        print("[PFM] Total force shape:", total_force.shape)
        return total_force, coeffs


class CheckpointedIntegratedMTAPFM_neighbours(nn.Module):
    """
    Hybrid neural/physics model for multi-agent trajectory prediction with full neighbor integration.

    Uses an LSTM-based encoder/decoder for trajectory forecasting,
    projects interaction coefficients, and corrects predictions with an interpretable Potential Field module.

    Args:
        input_size (int): Dimension of input positions (typically 2 for x, y).
        hidden_size (int): LSTM hidden state dimensionality.
        num_layers (int): Number of LSTM layers.
        target_avg_speed (float): Reference speed for enforcing agent motion constraints.
        speed_tolerance (float): Allowed deviation fraction from target speed.
        num_agents (int): Number of agent embeddings for the force coefficient module.
    """

    def __init__(self, input_size=2, hidden_size=64, num_layers=2,
                 target_avg_speed=4.087, speed_tolerance=0.15, num_agents=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_embed = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.coeff_projector = nn.Linear(hidden_size, 3)
        self.output = nn.Linear(hidden_size, input_size)
        self.pfm = PotentialField(num_agents)
        if target_avg_speed is None:
            raise ValueError("target_avg_speed must be provided")
        self.target_avg_speed = target_avg_speed
        self.speed_tolerance = speed_tolerance
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)
        self.dt = 0.1

    def forward_encoder(self, x):
        """
        Runs the LSTM encoder over the input embedding sequences.

        Args:
            x (Tensor): Embedded history sequences, shape [B * A * ent, H, D].

        Returns:
            output (Tensor), (h_n, c_n): LSTM outputs and hidden/cell state.
        """
        return self.encoder(x)

    def forward_decoder(self, x, hx):
        """
        Runs a step of the LSTM decoder.

        Args:
            x (Tensor): Embedded decoder input, shape [B * A * ent, 1, D].
            hx (tuple): Previous hidden and cell state.

        Returns:
            output (Tensor), (h_n, c_n): LSTM outputs and updated hidden/cell state.
        """
        return self.decoder(x, hx)

    def forward(self, history_neighbors, goal):
        """
        Main forward pass for trajectory forecasting and physical correction.
        Args:
            history_neighbors (Tensor): [B, A, ent, H, 2]
            goal (Tensor): [B, A, ent, 2] or [B, A, 2]
        Returns:
            adjusted_preds (Tensor), decoded_preds (Tensor), coeff_mean (Tensor), coeff_var (Tensor)
        """
        # --- FIX: Always extract shape from input chunk ---
        B, A, ent, H, D = history_neighbors.shape
        device = history_neighbors.device

        print("[MODEL] Forward pass started........")
        print("[MODEL] history_neighbors shape:", history_neighbors.shape)
        print("[MODEL] goal original shape:", goal.shape)
        print("[MODEL] Sample history (agent 0):", history_neighbors[0, 0, 0, :, :])
        print("[MODEL] Sample neighbor history (agent 0):", history_neighbors[0, 0, 1, :, :])

        # --- FIX: Expand goal only if needed ---
        if goal.shape == (B, A, D):
            goal = goal.unsqueeze(2).expand(B, A, ent, D).contiguous()
        elif goal.shape == (B, A, ent, D):
            pass  # Already correct shape
        else:
            raise AssertionError(f"Goal shape mismatch after expansion, got {goal.shape}, expected {(B, A, ent, D)}")

        hist_flat = history_neighbors.reshape(B * A * ent, H, D)
        emb = self.input_embed(hist_flat)
        _, (h_n, c_n) = torch.utils.checkpoint.checkpoint(self.forward_encoder, emb)

        h_top = h_n[-1].reshape(B, A, ent, self.hidden_size)
        coeffs = self.coeff_projector(h_top)

        pred_len = 12
        pred_flat = torch.zeros(B * A * ent, pred_len, D, device=device)
        hx = (h_n, c_n)

        history_flat = history_neighbors.view(B * A * ent, H, D)
        last_pos = torch.zeros(B * A * ent, 1, D, device=device)

        for idx in range(B * A * ent):
            for t in range(H - 1, -1, -1):
                pos = history_flat[idx, t]
                if not torch.all(pos == 0):
                    last_pos[idx] = pos
                    break
            else:
                last_pos[idx] = torch.zeros(D, device=device)

        # Decoder Loop
        for t in range(pred_len):
            if t == 0:
                decoder_in = last_pos.clone()
                print(f"[MODEL DEBUG] t={t} decoder input shape: {decoder_in.shape}")
            else:
                decoder_in = pred_flat[:, t - 1:t].clone()

            dec_emb = self.input_embed(decoder_in)
            h, c = hx
            hx = (h.contiguous(), c.contiguous())
            out, hx = torch.utils.checkpoint.checkpoint(self.forward_decoder, dec_emb, hx)
            step_out = self.output(out.squeeze(1))

            mean_step_out = step_out.mean(dim=0)
            print(f"step_out t={t} mean: {mean_step_out.tolist()}")
            print(f"step_out t={t} stats - mean: {step_out.mean().item()}, std: {step_out.std().item()}, min: {step_out.min().item()}, max: {step_out.max().item()}")

            if t == 0:
                pred_flat[:, t] = last_pos.squeeze(1) + step_out
                print("=== DEBUG timestep 0 ===")
                print("pred_flat sample:\n", pred_flat[:10, t])    # first 10 agents
                print("last_pos sample:\n", last_pos[:10])         # first 10 agents
                print("last_pos shape before squeeze:", last_pos.shape)
                print("last_pos shape after squeeze:", last_pos.squeeze(1).shape)
            else:
                pred_flat[:, t] = pred_flat[:, t - 1] + step_out

        decoded_preds = pred_flat.reshape(B, A, ent, pred_len, D)
        adjusted_preds = torch.zeros_like(decoded_preds)
        current_pos = last_pos.clone().reshape(B, A, ent, D)

        coeff_list = []
        for t in range(pred_len):
            print(f"[MODEL] Integration t={t}")
            for idx in range(ent):
                pos = current_pos[:, :, idx]
                pred = decoded_preds[:, :, idx, t]
                goal_vec = goal[:, :, idx]
                coeff = coeffs[:, :, idx]

                neighbors_idxs = [i for i in range(ent) if i != idx]

                neighbors_pos = torch.stack([current_pos[:, :, j] for j in neighbors_idxs], dim=2) if neighbors_idxs else torch.empty(B, A, 0, D, device=device)

                force, coeff_upd = self.pfm(pos, pred, neighbors_pos, goal_vec, coeff)

                if t == 0:
                    new_pos = pos + force
                else:
                    new_pos = adjusted_preds[:, :, idx, t - 1] + force

                if t > 0:
                    prev_pos = adjusted_preds[:, :, idx, t - 1]
                    disp = new_pos - prev_pos
                    speed = torch.norm(disp, dim=-1, keepdim=True)
                    clipped_speed = torch.clamp(speed, self.min_speed, self.max_speed)
                    adj_disp = disp / (speed + 1e-8) * clipped_speed
                    new_pos = prev_pos + adj_disp

                adjusted_preds[:, :, idx, t] = new_pos
                if idx == 0:
                    coeff_list.append(coeff_upd)

            current_pos = adjusted_preds[:, :, :, t].clone()

        coeff_stack = torch.stack(coeff_list)
        coeff_mean = coeff_stack.mean()
        coeff_var = coeff_stack.var(unbiased=False)

        print("[MODEL] Forward pass finished.")
        return adjusted_preds, decoded_preds, coeff_mean, coeff_var