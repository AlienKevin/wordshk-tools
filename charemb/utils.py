import torch

def normalize(vec):
    norm = torch.norm(vec)
    return vec / norm if norm != 0 else vec
