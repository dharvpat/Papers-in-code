import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from utils.positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding and Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Encoder and Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # Final linear layer for output
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        # Embedding and Positional Encoding for the source and target sequences
        src_embedded = self.positional_encoding(self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)))
        tgt_embedded = self.positional_encoding(self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)))
        
        # Pass through encoder and decoder
        encoder_output = self.encoder(src_embedded, src_mask)
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
        
        # Generate final output (logits before softmax)
        output = self.output_layer(decoder_output)
        return output