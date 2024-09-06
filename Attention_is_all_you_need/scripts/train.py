import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer
from model.transformer import Transformer
from utils.data_loader import load_data, get_data_loaders
from utils.loss import compute_loss
from config.config import Config

def train():
    # Load configuration
    cfg = Config()

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    input_data, target_data = load_data(cfg.input_file_path, cfg.target_file_path, tokenizer)

    # Prepare data loaders
    train_loader = get_data_loaders(input_data, target_data, tokenizer, batch_size=cfg.batch_size)

    # Initialize model
    model = Transformer(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout
    )
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0
        for batch_idx, (input_seqs, target_seqs) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_seqs)
            
            # Compute loss
            loss = compute_loss(outputs, target_seqs)
            total_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if batch_idx % cfg.log_interval == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), cfg.model_save_path)
    print(f"Model saved to {cfg.model_save_path}")

if __name__ == "__main__":
    train()