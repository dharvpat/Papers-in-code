import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    def __init__(self, base_model, embedding_dim=128):
        super(ContrastiveModel, self).__init__()
        self.encoder = base_model
        self.embedding_dim = embedding_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_dim)
        )
        # Remove the last fully connected layer from the base model
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        features = self.encoder(x)
        embeddings = self.projection_head(features)
        embeddings = F.normalize(embeddings, dim=1)  # Normalize embeddings
        return embeddings