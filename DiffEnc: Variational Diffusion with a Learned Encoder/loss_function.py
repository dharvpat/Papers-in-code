import torch

def diffusion_loss(x, predicted, lambda_reg=1e-5):
    mse_loss = torch.mean((x - predicted) ** 2)
    l2_reg = torch.norm(predicted, p=2)
    return mse_loss + lambda_reg * l2_reg