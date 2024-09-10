import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.size = size

    def forward(self, x, target):
        # Apply label smoothing
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)

def calculate_loss(predictions, targets, padding_idx):
    # Compute the standard cross-entropy loss while ignoring padding tokens
    loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)
    loss = loss_fn(predictions, targets)
    return loss