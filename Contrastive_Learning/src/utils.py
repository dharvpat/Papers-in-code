import torch

def save_checkpoint(state, filename="contrastive_model.pth"):
    torch.save(state, filename)
    print(f"Model checkpoint saved at {filename}")