import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import time

# Dataset Class
class PFM_TrajectoryDataset_Zara(torch.utils.data.Dataset):
    def __init__(self, file_path, history_len=8, prediction_len=12):
        self.data = self.load_data(file_path)
        self.history_len = history_len
        self.prediction_len = prediction_len
        # Create a list of valid frame indices that have enough history and future data
        self.valid_frames = self._get_valid_frames()

    def load_data(self, file_path):
        data = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 4:  # Ensure valid line
                    frame, agent, x, y = map(float, parts)
                    frame, agent = int(frame/10), int(agent)
                    if frame not in data:
                        data[frame] = {}
                    data[frame][agent] = torch.tensor([x, y], dtype=torch.float32)
        return data

    def _get_valid_frames(self):
        """Get frames that have sufficient history and future data"""
        all_frames = sorted(self.data.keys())
        valid_frames = []

        for frame in all_frames:
            # Check if we have enough history and future frames
            history_start = frame - self.history_len + 1
            future_end = frame + self.prediction_len

            # Ensure we have data for the required time range
            if history_start >= min(all_frames) and future_end <= max(all_frames):
                valid_frames.append(frame)

        return valid_frames

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, idx):
        frame = self.valid_frames[idx]

        # Get all agents present at this frame
        if frame not in self.data:
            # Return empty tensors if no data
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
            # Fill history (going backwards from current frame)
            for t in range(self.history_len):
                hist_frame = frame - (self.history_len - 1 - t)  # Fixed indexing
                if hist_frame in self.data and agent in self.data[hist_frame]:
                    history[i, t] = self.data[hist_frame][agent]
                # If no data available, position remains zero (padding)

            # Fill future (going forwards from next frame)
            for t in range(self.prediction_len):
                fut_frame = frame + t + 1  # Start from next frame
                if fut_frame in self.data and agent in self.data[fut_frame]:
                    future[i, t] = self.data[fut_frame][agent]
                # If no data available, position remains zero (padding)

            # Extract goal from the last timestep of future trajectory
            # Find the last non-zero position in future, or use the last timestep
            non_zero_mask = torch.any(future[i] != 0, dim=1)
            if non_zero_mask.any():
                last_valid_idx = torch.where(non_zero_mask)[0][-1]
                goals[i] = future[i, last_valid_idx]
            else:
                # If no future data, use current position as goal
                goals[i] = self.data[frame][agent]

        # Collect neighbors for each agent at the current frame
        neighbors_list = []
        for i, agent in enumerate(agents):
            # Get positions of all other agents at the current frame
            agent_neighbors = []
            for other_agent in self.data[frame]:
                if other_agent != agent:
                    agent_neighbors.append(self.data[frame][other_agent])

            if agent_neighbors:
                neighbors_tensor = torch.stack(agent_neighbors)
            else:
                # If no neighbors, create a dummy neighbor at origin
                neighbors_tensor = torch.zeros(1, 2)

            neighbors_list.append(neighbors_tensor)

        # Pad neighbors to have the same number for all agents
        if neighbors_list:
            max_neighbors = max(n.shape[0] for n in neighbors_list)
            padded_neighbors = torch.zeros(num_agents, max_neighbors, 2)

            for i, neighbor_tensor in enumerate(neighbors_list):
                padded_neighbors[i, :neighbor_tensor.shape[0]] = neighbor_tensor

            neighbors = padded_neighbors
        else:
            neighbors = torch.zeros(num_agents, 1, 2)

        return history, future, neighbors, goals