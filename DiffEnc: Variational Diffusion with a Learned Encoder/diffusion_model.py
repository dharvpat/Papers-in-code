import torch
import torch.nn as nn
from loss_function import diffusion_loss

class DiffusionModel(nn.Module):
    def __init__(self, encoder, decoder,device):
        super(DiffusionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def training_step(self, batch):
        x, _ = batch
        x = x.to(self.device)
        predicted = self(x)
        loss = diffusion_loss(x, predicted)
        return loss