import torch
import torch.nn as nn

class PotentialField(nn.Module):
    def __init__(self, goal=(4.2, 4.2), k_goal=1.0, k_pred=0.5, k_repulse=1.0, repulsion_radius=0.5):
        super().__init__()
        # Safely convert goal to tensor of shape [1, 2]
        if isinstance(goal, torch.Tensor):
            if goal.dim() == 1:
                goal = goal.unsqueeze(0)
            elif goal.dim() != 2 or goal.shape[1] != 2:
                raise ValueError("goal tensor must have shape [*,2]")
        else:
            goal = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)
        self.register_buffer('goal', goal)
        self.k_goal = k_goal
        self.k_pred = k_pred
        self.k_repulse = k_repulse
        self.repulsion_radius = repulsion_radius

    def forward(self, pos, predicted, neighbors, goal, coeffs=None):
        # Ensure goal matches pos shape for broadcasting
        if goal.shape[-1] > 2:
            goal = goal[..., :2]
        if goal.shape != pos.shape:
            goal = goal.expand_as(pos)

        Fg = self.k_goal * (goal - pos)
        Fp = self.k_pred * (predicted - pos)
        diffs = pos.unsqueeze(2) - neighbors
        dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
        mask = (dists < self.repulsion_radius).float()

        kr = self.k_repulse
        if coeffs is not None:
            kr = coeffs[..., 2:3]

        Fr = kr * (diffs / dists.pow(2)) * mask
        Fr = Fr.sum(dim=2)

        return Fg + Fp + Fr, torch.zeros_like(pos)

class PFMOnlyLearnable(nn.Module):
    def __init__(self, goal=(4.2, 4.2), target_avg_speed=0.027, speed_tolerance=0.15, 
                 num_agents=1000, dt=0.1, pred_len=12):
        super().__init__()
        if isinstance(goal, torch.Tensor):
            if goal.ndim == 1 and goal.shape[0] == 2:
                goal = goal.unsqueeze(0)  # from (2,) to (1,2)
            elif goal.ndim == 2 and goal.shape[1] == 2:
                pass  # OK
            else:
                raise ValueError(f"Expected goal tensor shape (*,2), got {goal.shape}")
        else:
            goal = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)
        self.register_buffer('goal', goal)

        # Handle target_avg_speed safely
        if isinstance(target_avg_speed, torch.Tensor):
            if target_avg_speed.numel() == 1:
                target_avg_speed = target_avg_speed.item()
            else:
                raise ValueError("target_avg_speed tensor must be scalar")
        self.target_avg_speed = float(target_avg_speed)

        self.k_goal = 1.0
        self.k_pred = 0.5
        self.k_repulse = 1.0
        self.repulsion_radius = 0.5
        self.speed_tolerance = speed_tolerance
        self.min_speed = self.target_avg_speed * (1 - speed_tolerance)
        self.max_speed = self.target_avg_speed * (1 + speed_tolerance)
        self.dt = dt
        self.pred_len = pred_len

        # No coeff embedding for non-learnable
        self.coeff_embedding = None
        self.pfm = PotentialField(goal=goal.squeeze(0))

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
            direction = disp / (sp + 1e-8)
            out[:, :, t] = cur + direction * sp_final
            cur = out[:, :, t].clone()
        return out

    def forward(self, history, neighbors, goal):
        # Ensure goal has shape (B, A, 2)
        if goal.dim() == 2:
            goal = goal.unsqueeze(1).expand(-1, history.shape[1], -1)

        B, A, H, _ = history.shape
        coeffs = None

        preds = torch.zeros(B, A, self.pred_len, 2, device=history.device)
        cur = history[:, :, -1, :].clone()

        for t in range(self.pred_len):
            if t == 0 and H >= 2:
                vel = history[:, :, -1, :] - history[:, :, -2, :]
                pred_input = (cur + vel).unsqueeze(2)
            elif t == 0:
                pred_input = cur.unsqueeze(2)
            else:
                pred_input = preds[:, :, t-1, :].unsqueeze(2)

            forces, _ = self.pfm(cur, pred_input, neighbors, goal, coeffs)
            cur = cur + forces * self.dt
            preds[:, :, t, :] = cur

        preds = self.apply_speed_constraints(preds, history[:, :, -1, :])
        return preds, None, None
