import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        # First linear transformation layer
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Activation function (ReLU)
        self.relu = nn.ReLU()
        
        # Second linear transformation layer
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Apply the first linear transformation followed by ReLU activation
        x = self.relu(self.linear1(x))
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Apply the second linear transformation
        x = self.linear2(x)
        
        return x