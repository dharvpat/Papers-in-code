import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),  # Batch normalization layer
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),  # Batch normalization layer
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU()
        )

        self.initialize_weights()  # Call weight initialization
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.encoder_layers(x)