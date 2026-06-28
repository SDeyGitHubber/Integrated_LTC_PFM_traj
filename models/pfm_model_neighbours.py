# import torch
# import torch.nn as nn


# class PotentialField(nn.Module):
#     def __init__(self, goal, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
#         super().__init__()
#         self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))
#         self.repulsion_radius = repulsion_radius
#         self.coeff_embedding = nn.Embedding(num_agents, 3)
#         self.coeff_embedding.weight.data.fill_(k_init)


#     def forward(self, pos, predicted, neighbors, goal, coeffs):
#         k1, k2, kr = coeffs[..., 0:1], coeffs[..., 1:2], coeffs[..., 2:3]
#         if predicted.dim() == 3:
#             Fp = k2 * (predicted - pos)
#         else:
#             # e.g., shape [B, A, 1, 2]
#             Fp = k2 * (predicted[:, :, 0, :] - pos)
        
#         Fg = k1 * (goal - pos)
#         diffs = pos.unsqueeze(2) - neighbors
#         dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
#         mask = (dists < self.repulsion_radius).float()
#         Fr = (kr.unsqueeze(2) * diffs / dists.pow(2) * mask).sum(dim=2)
#         return Fg + Fp + Fr, coeffs


# # === New PFMOnlyModel (replaces LSTM-based model) ===
# class PFMOnlyModel(nn.Module):
#     def __init__(self, goal=(4.2, 4.2), target_avg_speed=4.087,
#                  speed_tolerance=0.15, num_agents=1000, dt=0.1):
#         super().__init__()
#         self.pfm = PotentialField(goal, num_agents)
#         if target_avg_speed is None:
#             raise ValueError("target_avg_speed required")
#         self.min_speed = target_avg_speed * (1 - speed_tolerance)
#         self.max_speed = target_avg_speed * (1 + speed_tolerance)
#         self.dt = dt


#     def apply_speed_constraints(self, preds, last_pos):
#         B, A, T, _ = preds.shape
#         out = preds.clone()
#         cur = last_pos.clone()
#         for t in range(T):
#             disp = out[:, :, t] - cur
#             sp = torch.norm(disp, dim=-1, keepdim=True)
#             nz = sp > 0
#             clipped = torch.clamp(sp, self.min_speed, self.max_speed)
#             sp_final = torch.where(nz, clipped, sp)
#             dir = disp / (sp + 1e-8)
#             out[:, :, t] = cur + dir * sp_final
#             cur = out[:, :, t].clone()
#         return out


#     def forward(self, history, neighbors, goal):
#         B, A, H, _ = history.shape
#         agent_ids = torch.arange(A).repeat(B, 1).to(history.device)
#         coeffs = self.pfm.coeff_embedding(agent_ids)


#         preds = torch.zeros(B, A, 12, 2, device=history.device)
#         cur = history[:, :, -1, :].clone()
#         cur_neighbors = neighbors.clone()
#         coeff_list = []


#         for t in range(12):
#             if t == 0 and H >= 2:
#                 vel = history[:, :, -1, :] - history[:, :, -2, :]
#                 pred_slice = (cur + vel).unsqueeze(2)
#             elif t == 0:
#                 pred_slice = cur.unsqueeze(2)
#             else:
#                 pred_slice = preds[:, :, t-1:t, :].clone()


#             forces_ego, cstep_ego = self.pfm(cur, pred_slice, cur_neighbors, goal, coeffs)
#             nextp_ego = cur + forces_ego * self.dt
#             preds[:, :, t, :] = nextp_ego
#             cur = nextp_ego.detach()  # detach to save memory


#             B_, A_, N, _ = cur_neighbors.shape
#             cur_neighbors_flat = cur_neighbors.view(B_ * A_ * N, 2)
#             pred_slice_neighbors = cur_neighbors_flat.unsqueeze(1)
#             coeffs_neighbors = coeffs.unsqueeze(2).repeat(1, 1, N, 1).view(B_ * A_ * N, 3)


#             # Represent neighbors of neighbors as an empty tensor to avoid memory load
#             neighbors_neighbors = torch.empty(0, 0, 2, device=cur_neighbors.device)


#             # Correctly expand goal by adding neighbors dimension (dim=2)
#             goal_expanded = goal.unsqueeze(2).expand(B_, A_, N, 2).contiguous().view(B_ * A_ * N, 2)


#             forces_neighbors, _ = self.pfm(
#                 pos=cur_neighbors_flat,
#                 predicted=pred_slice_neighbors,
#                 neighbors=neighbors_neighbors,  # empty tensor here
#                 goal=goal_expanded,
#                 coeffs=coeffs_neighbors
#             )


#             nextp_neighbors = cur_neighbors_flat + forces_neighbors * self.dt
#             cur_neighbors = nextp_neighbors.view(B_, A_, N, 2).clone().detach()  # detach for memory efficiency


#             coeff_list.append(cstep_ego)


#         preds = self.apply_speed_constraints(preds, history[:, :, -1, :])
#         stack = torch.stack(coeff_list, dim=0)
#         return preds, stack.mean(), stack.var(unbiased=False)
    



    # INITIAL CODE
#     class PotentialField(nn.Module):
#     def __init__(self, goal, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
#         super().__init__()
#         self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))
#         self.repulsion_radius = repulsion_radius
#         self.coeff_embedding = nn.Embedding(num_agents, 3)
#         self.coeff_embedding.weight.data.fill_(k_init)
#     def forward(self, pos, predicted, neighbors, goal, coeffs):
#         k1, k2, kr = coeffs[...,0:1], coeffs[...,1:2], coeffs[...,2:3]
#         Fg = k1 * (goal - pos)
#         Fp = k2 * (predicted[:,:,0,:] - pos)
#         diffs = pos.unsqueeze(2) - neighbors
#         dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
#         mask = (dists < self.repulsion_radius).float()
#         Fr = (kr.unsqueeze(2) * diffs / dists.pow(2) * mask).sum(dim=2)
#         return Fg + Fp + Fr, coeffs
# # === New PFMOnlyModel (replaces LSTM-based model) ===
# class PFMOnlyModel(nn.Module):
#     def __init__(self, goal=(4.2,4.2), target_avg_speed=4.087,
#                  speed_tolerance=0.15, num_agents=1000, dt=0.1):
#         super().__init__()
#         self.pfm = PotentialField(goal, num_agents)
#         if target_avg_speed is None:
#             raise ValueError("target_avg_speed required")
#         self.min_speed = target_avg_speed * (1 - speed_tolerance)
#         self.max_speed = target_avg_speed * (1 + speed_tolerance)
#         self.dt = dt
#     def apply_speed_constraints(self, preds, last_pos):
#         B,A,T,_ = preds.shape
#         out = preds.clone()
#         cur = last_pos.clone()
#         for t in range(T):
#             disp = out[:,:,t] - cur
#             sp = torch.norm(disp, dim=-1, keepdim=True)
#             nz = sp>0
#             clipped = torch.clamp(sp, self.min_speed, self.max_speed)
#             sp_final = torch.where(nz, clipped, sp)
#             dir = disp/(sp+1e-8)
#             out[:,:,t] = cur + dir*sp_final
#             cur = out[:,:,t].clone()
#         # print(out{:,:,-2},out{:,:,-1},torch.norm(out{:,:,-2} - out{:,:,-1}, dim=-1, keepdim=True))    #RKL Add
#         return out
#     def forward(self, history, neighbors, goal):
#         B,A,H,_ = history.shape
#         agent_ids = torch.arange(A).repeat(B,1).to(history.device)
#         coeffs = self.pfm.coeff_embedding(agent_ids) #RKL1 cgange agent ids to history
#         preds = torch.zeros(B,A,12,2,device=history.device)
#         cur = history[:,:,-1,:].clone()
#         coeff_list=[]
#         for t in range(12):
#             if t==0 and H>=2:
#                 vel = history[:,:,-1,:] - history[:,:,-2,:]
#                 pred_slice = (cur+vel).unsqueeze(2)
#             elif t==0:
#                 pred_slice = cur.unsqueeze(2)
#             else:
#                 pred_slice = preds[:,:,t-1:t,:].clone()
#             forces, cstep = self.pfm(cur, pred_slice, neighbors, goal, coeffs) #RKL1 the current position i spred_slice, the future prediction should come from the neural network
#             # RKL1 move neighbours by 1 step using potential field only
#             nextp = cur + forces*self.dt
#             preds[:,:,t] = nextp
#             cur = nextp.clone()
#             coeff_list.append(cstep)
#         preds = self.apply_speed_constraints(preds, history[:,:,-1,:])
#         stack = torch.stack(coeff_list,dim=0)
#         return preds, stack.mean(), stack.var(unbiased=False)
# 

import torch
import torch.nn as nn


class PotentialField(nn.Module):
    """
    Physics-inspired module that computes social forces on each agent entity.
    """
    def __init__(self, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
        super().__init__()
        self.repulsion_radius = repulsion_radius
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)

    def forward(self, pos, predicted, neighbors, goal, coeffs):
        """
        Computes total potential field force.

        Args:
            pos: Current positions - shape can be [B, A, 2] or [N_flat, 2]
            predicted: Predicted positions - shape [B, A, 1, 2] or [N_flat, 1, 2]
            neighbors: Neighbor positions - shape [B, A, N, 2] or empty
            goal: Goal positions - shape [B, A, 2] or [N_flat, 2]
            coeffs: Force coefficients - shape [B, A, 3] or [N_flat, 3]

        Returns:
            total_force: Net force vector with same leading dims as pos
            coeffs: Same coefficients
        """
        k1, k2, kr = coeffs[..., 0:1], coeffs[..., 1:2], coeffs[..., 2:3]

        # Goal attraction
        Fg = k1 * (goal - pos)

        # Prediction attraction - handle various input shapes
        if predicted.dim() == 2:
            Fp = k2 * (predicted - pos)
        elif predicted.dim() == 3:
            Fp = k2 * (predicted.squeeze(1) - pos)
        elif predicted.dim() == 4:
            Fp = k2 * (predicted[:, :, 0, :] - pos)
        else:
            Fp = k2 * (predicted - pos)

        # Neighbor repulsion
        if neighbors.numel() == 0 or neighbors.size(-2) == 0:
            Fr = torch.zeros_like(pos)
        else:
            # Only compute if neighbors exist
            if pos.dim() == 2:
                # Flattened case: no repulsion (neighbors already accounted for at ego level)
                Fr = torch.zeros_like(pos)
            else:
                diffs = pos.unsqueeze(2) - neighbors
                dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
                mask = (dists < self.repulsion_radius).float()
                Fr = (kr.unsqueeze(2) * diffs / dists.pow(2) * mask).sum(dim=2)

        return Fg + Fp + Fr, coeffs


class PFMOnlyModel(nn.Module):
    """
    Pure Potential Field Model for multi-agent trajectory prediction.
    """
    def __init__(self, target_avg_speed=4.087, speed_tolerance=0.15,
                 num_agents=1000, dt=0.1):
        super().__init__()
        self.pfm = PotentialField(num_agents)
        if target_avg_speed is None:
            raise ValueError("target_avg_speed required")
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)
        self.dt = dt

    def apply_speed_constraints(self, preds, last_pos):
        B, A, T, _ = preds.shape
        out = preds.clone()
        cur = last_pos.clone()
        for t in range(T):
            disp = out[:, :, t] - cur
            sp = torch.norm(disp, dim=-1, keepdim=True)
            nz = sp > 0
            clipped = torch.clamp(sp, self.min_speed, self.max_speed)
            sp_final = torch.where(nz, clipped, sp)
            dir = disp / (sp + 1e-8)
            out[:, :, t] = cur + dir * sp_final
            cur = out[:, :, t].clone()
        return out

    def forward(self, history_neighbors, goal):
        """
        Args:
            history_neighbors: [B, A, N_entities, H, 2] where N_entities = 13 (1 ego + 12 neighbors)
            goal: [B, A, N_entities, 2]

        Returns:
            preds: [B, A, 12, 2]
            coeff_mean, coeff_var: scalars
        """
        B, A, N_entities, H, _ = history_neighbors.shape
        device = history_neighbors.device

        # Expand goal if needed
        if goal.dim() == 3 and goal.shape[2] == 2:
            goal = goal.unsqueeze(2).expand(B, A, N_entities, 2).contiguous()

        # Extract ego and neighbors
        ego_history = history_neighbors[:, :, 0, :, :]  # [B, A, H, 2]
        neighbor_history = history_neighbors[:, :, 1:, :, :]  # [B, A, 12, H, 2]

        # Get last positions
        cur_ego = ego_history[:, :, -1, :].clone()  # [B, A, 2]
        cur_neighbors = neighbor_history[:, :, :, -1, :].clone()  # [B, A, 12, 2]
        N = cur_neighbors.shape[2]

        # Get coefficients
        agent_ids = torch.arange(A, device=device).unsqueeze(0).expand(B, A)
        coeffs = self.pfm.coeff_embedding(agent_ids)  # [B, A, 3]

        preds = torch.zeros(B, A, 12, 2, device=device)
        coeff_list = []

        for t in range(12):
            # Prediction slice for ego
            if t == 0 and H >= 2:
                vel = ego_history[:, :, -1, :] - ego_history[:, :, -2, :]
                pred_slice = (cur_ego + vel).unsqueeze(2)
            elif t == 0:
                pred_slice = cur_ego.unsqueeze(2)
            else:
                pred_slice = preds[:, :, t-1:t, :].clone()

            # Update ego
            ego_goal = goal[:, :, 0, :]
            forces_ego, cstep_ego = self.pfm(
                cur_ego, pred_slice, cur_neighbors, ego_goal, coeffs
            )
            nextp_ego = cur_ego + forces_ego * self.dt
            preds[:, :, t, :] = nextp_ego
            cur_ego = nextp_ego.detach()

            # Update neighbors (simplified - just goal attraction, no inter-neighbor repulsion)
            if N > 0:
                # Shape check
                assert cur_neighbors.shape == (B, A, N, 2), f"cur_neighbors shape mismatch: {cur_neighbors.shape}"

                # Flatten for vectorized computation
                cur_neighbors_flat = cur_neighbors.reshape(B * A * N, 2)

                # Compute simple goal attraction for neighbors (no complex repulsion)
                goal_neighbors = goal[:, :, 1:N+1, :].reshape(B * A * N, 2)
                coeffs_neighbors = coeffs[:, :, 0:1].unsqueeze(2).repeat(1, 1, N, 1).reshape(B * A * N, 1)

                # Simple goal-directed motion
                goal_force = coeffs_neighbors * (goal_neighbors - cur_neighbors_flat)
                nextp_neighbors_flat = cur_neighbors_flat + goal_force * self.dt

                # Reshape back
                cur_neighbors = nextp_neighbors_flat.reshape(B, A, N, 2).detach()

            coeff_list.append(cstep_ego)

        preds = self.apply_speed_constraints(preds, ego_history[:, :, -1, :])
        stack = torch.stack(coeff_list, dim=0)
        return preds, stack.mean(), stack.var(unbiased=False)
