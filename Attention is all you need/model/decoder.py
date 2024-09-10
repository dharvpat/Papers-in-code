import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward
from utils.norm import LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, src_mask, tgt_mask):
        # Self-Attention with residual connection and layer normalization
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-Attention with residual connection and layer normalization
        tgt2 = self.cross_attn(tgt, memory, memory, src_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward with residual connection and layer normalization
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
    
    def forward(self, tgt, memory, src_mask, tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        return self.norm(tgt)
