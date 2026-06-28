import torch

def collate_fn(batch):
    """Collate function for batches that include goal"""
    max_agents = max(sample[0].shape[0] for sample in batch)
    # history
    max_neighbors = max(sample[2].shape[1] for sample in batch)  # neighbors
    hist_len = batch[0][0].shape[1]
    fut_len = batch[0][1].shape[1]

    padded_histories = []
    padded_futures = []
    padded_neighbors = []
    padded_goals = []

    for sample in batch:
        history, future, neighbors, goal = sample
        A, H, D = history.shape
        N = neighbors.shape[1]

        padded_hist = torch.zeros(max_agents, hist_len, D)
        padded_fut = torch.zeros(max_agents, fut_len, D)
        padded_neigh = torch.zeros(max_agents, max_neighbors, D)
        padded_goal = torch.zeros(max_agents, D)

        padded_hist[:A] = history
        padded_fut[:A] = future
        padded_neigh[:A, :N] = neighbors
        padded_goal[:A] = goal

        padded_histories.append(padded_hist)
        padded_futures.append(padded_fut)
        padded_neighbors.append(padded_neigh)
        padded_goals.append(padded_goal)

    return (
        torch.stack(padded_histories),   # [B, A, hist_len, D]
        torch.stack(padded_futures),     # [B, A, fut_len, D]
        torch.stack(padded_neighbors),   # [B, A, max_neighbors, D]
        torch.stack(padded_goals)        # [B, A, D]
    )

# FIXED: Added goals to test data (4th element in each tuple)
test_batch = [
    (torch.rand(3, 8, 2), torch.rand(3, 12, 2), torch.rand(3, 5, 2), torch.rand(3, 2)),  # 3 agents with goals
    (torch.rand(2, 8, 2), torch.rand(2, 12, 2), torch.rand(2, 3, 2), torch.rand(2, 2)),  # 2 agents with goals
    (torch.rand(4, 8, 2), torch.rand(4, 12, 2), torch.rand(4, 6, 2), torch.rand(4, 2))   # 4 agents with goals
]

# Now this should work correctly
hist, fut, neigh, goals = collate_fn(test_batch)  # Note: also need to unpack goals here
print(f"History shape: {hist.shape}")     # Will be [3, 4, 8, 2]
print(f"Future shape: {fut.shape}")       # Will be [3, 4, 12, 2]
print(f"Neighbors shape: {neigh.shape}")  # Will be [3, 4, 6, 2]
print(f"Goals shape: {goals.shape}")      # Will be [3, 4, 2]