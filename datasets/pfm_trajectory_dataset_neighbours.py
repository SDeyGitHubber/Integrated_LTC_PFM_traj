# dataloader.py

import torch
from torch.utils.data import Dataset
import os # Added os import for file path checking/mock

class PFM_TrajectoryDataset_neighbours(Dataset):
    """
    Dataset class for loading trajectory data, preparing history, future,
    and neighbor information for use in models like MTA-PFM.
    """
    def __init__(self, file_path, history_len=8, prediction_len=12, max_neighbors=12, debug=False):
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.max_neighbors = max_neighbors
        self.debug = debug
        self.data = self.load_data(file_path)
        self.valid_frames = self._get_valid_frames()

    def load_data(self, file_path):
        data = {}
        # Simple check for file existence
        if not os.path.exists(file_path):
             print(f"[DL-LOAD][WARN] File not found: {file_path}. Returning empty data.")
             return {}

        print(f"[DL-LOAD] Loading data file: {file_path}")
        with open(file_path, "r") as file:
            for ln, line in enumerate(file, start=1):
                parts = line.strip().split(",")
                if len(parts) == 4:
                    try:
                        frame, agent, x, y = map(float, parts)
                        frame, agent = int(frame), int(agent)
                        if frame not in data:
                            data[frame] = {}
                        data[frame][agent] = torch.tensor([x, y], dtype=torch.float32)
                    except ValueError:
                         print(f"[DL-LOAD][WARN] Line {ln}: Non-numeric values found: {line.strip()}")
                else:
                    print(f"[DL-LOAD][WARN] Line {ln}: bad format (expected 4 parts): {line.strip()}")
        return data

    def _get_valid_frames(self):
        """Finds frames that have a complete history and future window."""
        if not self.data:
            return []

        all_frames = sorted(self.data.keys())
        if not all_frames:
             return []

        min_frame = min(all_frames)
        max_frame = max(all_frames)
        valid_frames = []
        for frame in all_frames:
            history_start = frame - self.history_len + 1
            future_end = frame + self.prediction_len
            if history_start >= min_frame and future_end <= max_frame:
                valid_frames.append(frame)
        return valid_frames

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, idx):
        frame = self.valid_frames[idx]

        # Define output dimensions for empty return
        N_entities = self.max_neighbors + 1 # Ego + Neighbors
        T_hist = self.history_len
        T_pred = self.prediction_len

        if frame not in self.data or not self.data[frame]:
            # Returns 6 empty tensors
            return (torch.zeros(0, N_entities, T_hist, 2),  # history_neighbors
                    torch.zeros(0, T_pred, 2),              # future (ego only)
                    torch.zeros(0, self.max_neighbors, T_hist, 2), # neighbor_histories
                    torch.zeros(0, 2),                      # goals (ego only)
                    torch.zeros(0, N_entities, 2),          # expanded_goals
                    torch.zeros(0, N_entities, T_pred, 2))  # all_futures (NEW)

        agents = list(self.data[frame].keys())
        num_agents = len(agents)

        history = torch.zeros(num_agents, T_hist, 2)
        future = torch.zeros(num_agents, T_pred, 2) # Ego future
        goals = torch.zeros(num_agents, 2) # Ego goal

        # 1. Gather Ego history, future, and goal
        for i, agent in enumerate(agents):
            for t in range(T_hist):
                hist_frame = frame - (T_hist - 1 - t)
                if hist_frame in self.data and agent in self.data[hist_frame]:
                    history[i, t] = self.data[hist_frame][agent]
            for t in range(T_pred):
                fut_frame = frame + t + 1
                if fut_frame in self.data and agent in self.data[fut_frame]:
                    future[i, t] = self.data[fut_frame][agent]

            # Define Ego Goal as the last valid point in the future
            non_zero_mask = torch.any(future[i] != 0, dim=1)
            if non_zero_mask.any():
                last_valid_idx = torch.where(non_zero_mask)[0][-1]
                goals[i] = future[i, last_valid_idx]
            else:
                goals[i] = self.data[frame][agent] # Default to current position if no future

        neighbor_histories = torch.zeros(num_agents, self.max_neighbors, T_hist, 2)
        neighbor_goals = torch.zeros(num_agents, self.max_neighbors, 2)
        neighbor_futures = torch.zeros(num_agents, self.max_neighbors, T_pred, 2) # NEW

        # 2. Gather Neighbor information
        for i, agent in enumerate(agents):
            ego_pos = self.data[frame][agent]
            other_agents_with_dist = []
            for other_agent in agents:
                if other_agent == agent:
                    continue
                other_pos = self.data[frame][other_agent]
                # Calculate distance at current frame
                dist = torch.norm(ego_pos - other_pos).item()
                other_agents_with_dist.append((other_agent, dist))

            # Select the closest neighbors
            other_agents_with_dist.sort(key=lambda x: x[1])
            other_agents = [x[0] for x in other_agents_with_dist[:self.max_neighbors]]

            for n_idx, neighbor in enumerate(other_agents):
                # Neighbor History
                for t in range(T_hist):
                    hist_frame = frame - (T_hist - 1 - t)
                    if hist_frame in self.data and neighbor in self.data[hist_frame]:
                        neighbor_histories[i, n_idx, t] = self.data[hist_frame][neighbor]

                # Neighbor Future (NEW)
                for t in range(T_pred):
                    fut_frame = frame + t + 1
                    if fut_frame in self.data and neighbor in self.data[fut_frame]:
                        neighbor_futures[i, n_idx, t] = self.data[fut_frame][neighbor]

                # Neighbor Goal (last valid point in neighbor future)
                neighbor_non_zero_mask = torch.any(neighbor_futures[i, n_idx] != 0, dim=1)
                if neighbor_non_zero_mask.any():
                    neighbor_last_valid_idx = torch.where(neighbor_non_zero_mask)[0][-1]
                    neighbor_goals[i, n_idx] = neighbor_futures[i, n_idx, neighbor_last_valid_idx]
                else:
                    # Default to current position if no future
                    if frame in self.data and neighbor in self.data[frame]:
                        neighbor_goals[i, n_idx] = self.data[frame][neighbor]
                    else:
                        neighbor_goals[i, n_idx] = torch.zeros(2)

        # 3. Mask invalid agents (agents with incomplete data)
        # Check if history and future for the ego agent are available.
        mask = torch.ones(num_agents, dtype=torch.bool)
        for i in range(history.shape[0]):
            # An agent is invalid if its own history OR future is all zeros
            if not torch.any(history[i]) or not torch.any(future[i]):
                mask[i] = False
            # NOTE: The original code included a check on neighbor_histories here,
            # which is often too strict (it masks agents if ALL neighbors are zero).
            # We remove that strict neighbor check to keep more agents, but you
            # might need to re-add it if your task requires valid neighbors for every ego.
            # if torch.any(torch.all(neighbor_histories[i] == 0, dim=(1, 2))):
            #     mask[i] = False

        history = history[mask]
        future = future[mask]
        goals = goals[mask]
        neighbor_histories = neighbor_histories[mask]
        neighbor_goals = neighbor_goals[mask]
        neighbor_futures = neighbor_futures[mask]  # NEW

        # 4. Final concatenation and expansion

        # history_neighbors: [N_valid, 1+N_max, T_hist, 2]
        ego_history = history.unsqueeze(1)
        history_neighbors = torch.cat((ego_history, neighbor_histories), dim=1)

        # all_futures: [N_valid, 1+N_max, T_pred, 2]
        ego_future = future.unsqueeze(1)
        all_futures = torch.cat((ego_future, neighbor_futures), dim=1)

        # expanded_goals: [N_valid, 1+N_max, 2]
        expanded_goals = torch.zeros(history_neighbors.shape[0], N_entities, 2)
        for i in range(goals.shape[0]):
            expanded_goals[i, 0, :] = goals[i] # Ego goal at index 0
            for j in range(self.max_neighbors):
                # Neighbor goals start at index 1
                expanded_goals[i, j + 1, :] = neighbor_goals[i, j]

        # Debug print statement
          if self.debug:
            print("[DATASET] __getitem__ output shapes:",
                "history_neighbors", history_neighbors.shape,
                "future (ego)", future.shape,
                "neighbor_histories", neighbor_histories.shape,
                "goals (ego)", goals.shape,
                "expanded_goals", expanded_goals.shape,
                "all_futures", all_futures.shape)

        # Returns 6 Tensors
        return history_neighbors, future, neighbor_histories, goals, expanded_goals, all_futures