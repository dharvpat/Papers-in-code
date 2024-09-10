import torch.optim as optim
import numpy as np

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # Update parameters and rate
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # Compute the learning rate with the Noam scheme
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * \
               min(step ** -0.5, step * self.warmup ** -1.5)

def get_std_opt(model):
    # A standard optimizer with the Noam scheme
    return NoamOpt(model_size=model.d_model, factor=2, warmup=4000,
                   optimizer=optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))