import torch
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for input_seq, target_seq in data_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            loss = compute_loss(outputs, target_seq)
            total_loss += loss.item()

            preds = outputs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target_seq.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(np.concatenate(all_targets), np.concatenate(all_preds))
    return avg_loss, accuracy

def compute_loss(outputs, targets):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Assuming padding index is 0
    return loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))