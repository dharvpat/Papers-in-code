import torch
import torch.nn.functional as F

def rpn_loss(cls_logits, bbox_preds, labels, bbox_targets):
    classification_loss = F.cross_entropy(cls_logits, labels)
    regression_loss = F.smooth_l1_loss(bbox_preds, bbox_targets, reduction='sum')
    return classification_loss + regression_loss