import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def get_faster_rcnn_model(num_classes):
    # Load a pre-trained model for feature extraction
    backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048

    # Define the anchor generator for the RPN
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Define the RoI align layer
    roi_pooler = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(7, 7)),
        nn.Flatten()
    )

    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model