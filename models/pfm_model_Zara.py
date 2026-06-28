import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# === Model definition ===
class PotentialField(nn.Module):
    def __init__(self, goal, num_agents=1000, k_init=1.0, repulsion_radius=0.5):
        super().__init__()
        self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))
        self.repulsion_radius = repulsion_radius
        self.coeff_embedding = nn.Embedding(num_agents, 3)
        self.coeff_embedding.weight.data.fill_(k_init)

    def forward(self, pos, predicted, neighbors, goal, coeffs):
        k1, k2, kr = coeffs[..., 0:1], coeffs[..., 1:2], coeffs[..., 2:3]
        Fg = k1 * (goal - pos)
        Fp = k2 * (predicted - pos)
        diffs = pos.unsqueeze(2) - neighbors
        dists = torch.norm(diffs, dim=-1, keepdim=True) + 1e-6
        mask = (dists < self.repulsion_radius).float()
        Fr = (kr.unsqueeze(2) * diffs / dists.pow(2) * mask).sum(dim=2)
        return Fg + Fp + Fr, coeffs


class PFMModel(nn.Module):
    def __init__(self, goal=(4.2, 4.2), target_avg_speed=4.087, speed_tolerance=0.15,
                 num_agents=1000, dt=0.1, pred_len=12):
        super().__init__()
        self.pfm = PotentialField(goal, num_agents)
        self.target_avg_speed = target_avg_speed
        self.speed_tolerance = speed_tolerance
        self.dt = dt
        self.pred_len = pred_len
        self.min_speed = target_avg_speed * (1 - speed_tolerance)
        self.max_speed = target_avg_speed * (1 + speed_tolerance)

    def apply_speed_constraints(self, preds, last_pos):
        out = preds.clone()
        cur = last_pos.clone()
        for t in range(preds.shape[2]):
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
        agent_ids = torch.arange(A).expand(B, A).to(history.device)
        coeffs = self.pfm.coeff_embedding(agent_ids)
        preds = torch.zeros(B, A, self.pred_len, 2, device=history.device)
        cur = history[:, :, -1].clone()
        for t in range(self.pred_len):
            if t == 0 and H > 1:
                vel = history[:, :, -1] - history[:, :, -2]
                pred_input = (cur + vel).unsqueeze(2)
            elif t == 0:
                pred_input = cur.unsqueeze(2)
            else:
                pred_input = preds[:, :, t-1:t]
            forces, _ = self.pfm(cur, pred_input, neighbors, goal, coeffs)
            next_pos = cur + forces * self.dt
            preds[:, :, t] = next_pos
            cur = next_pos.clone()
        preds = self.apply_speed_constraints(preds, history[:, :, -1])
        return preds, None, None