import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    # Compute the dot products between query and key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    
    # Apply the mask (if provided) by setting masked positions to a large negative value
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply the softmax function to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Multiply the attention weights by the value to get the output
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Linear layers to project input to query, key, and value vectors
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        # Linear layer to project concatenated output back to the original d_model size
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Apply linear layers and split into multiple heads
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(query, key, value, mask)
        
        # Concatenate the heads and apply the final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attention_output)
        
        return output, attention_weights