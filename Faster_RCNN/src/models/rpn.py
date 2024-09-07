import torch.nn as nn

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RegionProposalNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg