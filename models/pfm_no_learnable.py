import torch
from torch import nn

class PotentialFieldNoLearnable(nn.Module):
    def __init__(self, goal, k_goal=1.0, k_pred=0.5, k_repulse=1.0, repulsion_radius=0.5):
        super().__init__()
        self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))
        self.k_goal = k_goal
        self.k_pred = k_pred
        self.k_repulse = k_repulse
        self.repulsion_radius = repulsion_radius

    def forward(self, pos, predicted, neighbors, goal, coeffs=None):
        # coeffs ignored; use fixed values
        Fg = self.k_goal * (goal - pos)                        # Goal attraction
        Fp = self.k_pred * (predicted[:, :, 0, :] - pos)       # Predicted movement attraction

        diffs = pos.unsqueeze(2) - neighbors                   # [B, A, max_neighbors, 2]
        dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
        mask = (dists < self.repulsion_radius).float()
        Fr = (self.k_repulse * diffs / dists.pow(2) * mask).sum(dim=2)
        # Ignore coeffs, just use fixed constants for each term
        return Fg + Fp + Fr, torch.zeros_like(pos)

class PFMOnlyModelNoLearnable(nn.Module):
    def __init__(self, goal=(4.2, 4.2), target_avg_speed=4.087,
                 speed_tolerance=0.15, dt=0.1,
                 k_goal=1.0, k_pred=0.5, k_repulse=1.0, repulsion_radius=0.5):
        super().__init__()
        self.pfm = PotentialFieldNoLearnable(goal, k_goal, k_pred, k_repulse, repulsion_radius)
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

    def forward(self, history, neighbors, goal):
        B, A, H, _ = history.shape
        preds = torch.zeros(B, A, 12, 2, device=history.device)
        cur = history[:, :, -1, :].clone()
        for t in range(12):
            if t == 0 and H >= 2:
                vel = history[:, :, -1, :] - history[:, :, -2, :]
                pred_slice = (cur + vel).unsqueeze(2)
            elif t == 0:
                pred_slice = cur.unsqueeze(2)
            else:
                pred_slice = preds[:, :, t-1:t, :].clone()
            forces, _ = self.pfm(cur, pred_slice, neighbors, goal, None)
            nextp = cur + forces * self.dt
            preds[:, :, t] = nextp
            cur = nextp.clone()
        preds = self.apply_speed_constraints(preds, history[:, :, -1, :])
        return preds, None, None
