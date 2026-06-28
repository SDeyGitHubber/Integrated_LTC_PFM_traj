import torch
from torch.utils.data import Dataset

class PFM_TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, history_len=8, prediction_len=12):
        self.data = self.load_data(file_path)
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.valid_frames = self._get_valid_frames()

    def load_data(self, file_path):
        data = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    frame, agent, x, y = map(float, parts)
                    frame, agent = int(frame), int(agent)
                    if frame not in data:
                        data[frame] = {}
                    data[frame][agent] = torch.tensor([x, y], dtype=torch.float32)
        return data

    def _get_valid_frames(self):
        all_frames = sorted(self.data.keys())
        valid_frames = []
        for frame in all_frames:
            history_start = frame - self.history_len + 1
            future_end = frame + self.prediction_len
            if history_start >= min(all_frames) and future_end <= max(all_frames):
                valid_frames.append(frame)
        return valid_frames

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, idx):
        frame = self.valid_frames[idx]
        if frame not in self.data:
            return (torch.zeros(0, self.history_len, 2),
                    torch.zeros(0, self.prediction_len, 2),
                    torch.zeros(0, 0, 2),
                    torch.zeros(0, 2))

        agents = list(self.data[frame].keys())
        num_agents = len(agents)
        if num_agents == 0:
            return (torch.zeros(0, self.history_len, 2),
                    torch.zeros(0, self.prediction_len, 2),
                    torch.zeros(0, 0, 2),
                    torch.zeros(0, 2))

        history = torch.zeros(num_agents, self.history_len, 2)
        future = torch.zeros(num_agents, self.prediction_len, 2)
        goals = torch.zeros(num_agents, 2)

        for i, agent in enumerate(agents):
            for t in range(self.history_len):
                hist_frame = frame - (self.history_len - 1 - t)
                if hist_frame in self.data and agent in self.data[hist_frame]:
                    history[i, t] = self.data[hist_frame][agent]
            for t in range(self.prediction_len):
                fut_frame = frame + t + 1
                if fut_frame in self.data and agent in self.data[fut_frame]:
                    future[i, t] = self.data[fut_frame][agent]
            non_zero_mask = torch.any(future[i] != 0, dim=1)
            if non_zero_mask.any():
                last_valid_idx = torch.where(non_zero_mask)[0][-1]
                goals[i] = future[i, last_valid_idx]
            else:
                goals[i] = self.data[frame][agent]

        neighbors_list = []
        for i, agent in enumerate(agents):
            agent_neighbors = []
            for other_agent in self.data[frame]:
                if other_agent != agent:
                    agent_neighbors.append(self.data[frame][other_agent])
            if agent_neighbors:
                neighbors_tensor = torch.stack(agent_neighbors)
            else:
                neighbors_tensor = torch.zeros(1, 2)
            neighbors_list.append(neighbors_tensor)

        if neighbors_list:
            max_neighbors = 12
            padded_neighbors = torch.zeros(num_agents, max_neighbors, 2)
            for i, neighbor_tensor in enumerate(neighbors_list):
                if neighbor_tensor.shape[0] > 0:
                    ego_pos = history[i, -1].unsqueeze(0)
                    dists = torch.norm(neighbor_tensor - ego_pos, dim=1)
                    sorted_idx = torch.argsort(dists)
                    sorted_neighbors = neighbor_tensor[sorted_idx]
                    top_neighbors = sorted_neighbors[:max_neighbors]
                    padded_neighbors[i, :top_neighbors.shape[0]] = top_neighbors
            neighbors = padded_neighbors
        else:
            neighbors = torch.zeros(num_agents, 1, 2)

        mask = torch.ones(history.shape[0], dtype=torch.bool)
        i = 0
        while i<history.shape[0]:
            for seq in range(history.shape[1]):
                if history[i,seq,0] == 0 and history[i,seq,1]==0:
                    mask[i] = False
            for seq in range(future.shape[1]):
                if future[i,seq,0] == 0 and future[i,seq,1]==0:
                    mask[i] = False
            i+=1
        history = history[mask]
        future = future[mask]
        neighbors = neighbors[mask]
        goals = goals[mask]

        return history, future, neighbors, goals