import torch
import torch.nn.functional as F
import torch.nn as nn

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        batch_size = labels.shape[0]
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # Mask to remove self-comparisons
        mask_self = torch.eye(batch_size, device=features.device).float()
        mask = mask * (1 - mask_self)
        
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = -mean_log_prob.mean()
        return loss