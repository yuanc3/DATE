import torch


def get_final_indices(topk_indices, timestamps, attn_weights, interval=20.0, max_frames=256):
    if len(topk_indices)<=max_frames:
        selected_timestamps = [timestamps[x] for x in topk_indices]
        return selected_timestamps, topk_indices
    
    frame_info = [(idx, timestamps[idx], attn_weights[idx]) for i, idx in enumerate(topk_indices)]
    selected_indices, selected_timestamps = [], []
    decay_ratio = 0.5

    while len(selected_indices) < max_frames:
        for idx, ts, attn in frame_info:
            if idx in selected_indices:
                continue
            if all(abs(ts - existing_ts) >= interval for existing_ts in selected_timestamps):
                selected_indices.append(idx)
                selected_timestamps.append(ts)
                if len(selected_indices) >= max_frames:
                    break
        interval *= decay_ratio

    return selected_timestamps, selected_indices


def tass_sampling(timestamps, attn_weights, topk=None, max_frames=256):
    if topk is not None:
        length = len(attn_weights)
        trg = sum(attn_weights)/length
        topk = len([x for x in attn_weights if x > trg]) # only keep the positive samples
    
    up = max_frames * 4
    topk = min(topk, up)
    attn_weights = torch.tensor(attn_weights)
    topk_indices = torch.topk(attn_weights, k=topk).indices.tolist()
    selected_timestamps, selected_indices = get_final_indices(topk_indices, timestamps, attn_weights, interval = 20, max_frames=max_frames)
    selected_timestamps = sorted(selected_timestamps)
    return selected_timestamps, selected_indices
