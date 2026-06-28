# import torch

# def collate_fn(batch, history_len=None, prediction_len=None, max_neighbors=None):
#     print(f"[COLLATE] === COLLATE FUNCTION DEBUG ===")
#     print(f"[COLLATE] Batch size: {len(batch)}")
#     print(f"[COLLATE] First sample type: {type(batch[0])}")
#     print(f"[COLLATE] First sample length: {len(batch[0])}")
    
#     # FIX: Updated to handle 5 return values from dataloader
#     histories, futures, neighbor_histories, goals, expanded_goals = zip(*batch)
#     print(f"[COLLATE] Unpacked 5 components successfully")
    
#     print(f"[COLLATE] Component shapes before stacking:")
#     print(f"[COLLATE]   histories[0]: {histories[0].shape}")
#     print(f"[COLLATE]   futures[0]: {futures[0].shape}")
#     print(f"[COLLATE]   neighbor_histories[0]: {neighbor_histories[0].shape}")
#     print(f"[COLLATE]   goals[0]: {goals[0].shape}")
#     print(f"[COLLATE]   expanded_goals[0]: {expanded_goals[0].shape}")
    
#     # Stack all components
#     history_batch = torch.stack(histories, dim=0)
#     future_batch = torch.stack(futures, dim=0)
#     neighbor_histories_batch = torch.stack(neighbor_histories, dim=0)
#     goals_batch = torch.stack(goals, dim=0)
#     expanded_goals_batch = torch.stack(expanded_goals, dim=0)
    
#     print(f"[COLLATE] Output shapes after stacking:")
#     print(f"[COLLATE]   history_batch: {history_batch.shape}")
#     print(f"[COLLATE]   future_batch: {future_batch.shape}")
#     print(f"[COLLATE]   neighbor_histories_batch: {neighbor_histories_batch.shape}")
#     print(f"[COLLATE]   goals_batch: {goals_batch.shape}")
#     print(f"[COLLATE]   expanded_goals_batch: {expanded_goals_batch.shape}")
#     print(f"[COLLATE] === END COLLATE DEBUG ===")
    
#     return history_batch, future_batch, neighbor_histories_batch, goals_batch, expanded_goals_batch

# collate.py

# collate.py

import torch

def pad_to_max_agent_dim(tensors):
    max_agents = max(t.shape[0] for t in tensors)
    padded_tensors = []
    for t in tensors:
        pad_size = max_agents - t.shape[0]
        if pad_size > 0:
            pad_shape = [pad_size] + list(t.shape[1:])
            padding = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
            padded_tensors.append(torch.cat([t, padding], dim=0))
        else:
            padded_tensors.append(t)
    return torch.stack(padded_tensors, dim=0)

def collate_fn(batch, history_len=None, prediction_len=None, max_neighbors=None):
    print(f"[COLLATE] === COLLATE FUNCTION DEBUG ===")
    print(f"[COLLATE] Batch size: {len(batch)}")
    print(f"[COLLATE] First sample type: {type(batch[0])}")
    print(f"[COLLATE] First sample length: {len(batch[0])}")

    histories, futures, neighbor_histories, goals, expanded_goals, all_futures = zip(*batch)  # Unpack 6 items
    print(f"[COLLATE] Unpacked 5 components successfully")

    print(f"[COLLATE] Component shapes before padding and stacking:")
    print(f"[COLLATE]   histories[0]: {histories[0].shape}")
    print(f"[COLLATE]   futures[0]: {futures[0].shape}")
    print(f"[COLLATE]   neighbor_histories[0]: {neighbor_histories[0].shape}")
    print(f"[COLLATE]   goals[0]: {goals[0].shape}")
    print(f"[COLLATE]   expanded_goals[0]: {expanded_goals[0].shape}")

    # Pad on agent dimension across batch
    history_batch = pad_to_max_agent_dim(histories)
    future_batch = pad_to_max_agent_dim(futures)
    neighbor_histories_batch = pad_to_max_agent_dim(neighbor_histories)
    goals_batch = pad_to_max_agent_dim(goals)
    expanded_goals_batch = pad_to_max_agent_dim(expanded_goals)
    all_futures_batch = pad_to_max_agent_dim(all_futures)  # NEW

    print(f"[COLLATE] Output shapes after padding and stacking:")
    print(f"[COLLATE]   history_batch: {history_batch.shape}")
    print(f"[COLLATE]   future_batch: {future_batch.shape}")
    print(f"[COLLATE]   neighbor_histories_batch: {neighbor_histories_batch.shape}")
    print(f"[COLLATE]   goals_batch: {goals_batch.shape}")
    print(f"[COLLATE]   expanded_goals_batch: {expanded_goals_batch.shape}")
    print(f"[COLLATE] === END COLLATE DEBUG ===")

    return history_batch, future_batch, neighbor_histories_batch, goals_batch, expanded_goals_batch, all_futures_batch
