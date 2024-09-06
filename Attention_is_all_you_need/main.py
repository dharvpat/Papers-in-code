import argparse
import torch
from torch.utils.data import DataLoader
from model.transformer import Transformer
from utils.data_loader import load_data, CustomDataset
from utils.optimizer import NoamOpt
from utils.loss import LabelSmoothingLoss
from utils.evaluation import evaluate_model
from scripts.train import train_model
from config.config import Config

def main(args):
    # Load configuration
    config = Config(args.config_file)
    # Load data
    train_data, val_data, test_data, vocab_size = load_data(config)
    # Create DataLoaders
    train_loader = DataLoader(CustomDataset(train_data), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(CustomDataset(val_data), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(CustomDataset(test_data), batch_size=config.batch_size, shuffle=False)
    # Initialize model
    model = Transformer(vocab_size=vocab_size, 
                        d_model=config.d_model, 
                        num_heads=config.num_heads, 
                        num_layers=config.num_layers, 
                        d_ff=config.d_ff)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize optimizer and learning rate scheduler
    optimizer = NoamOpt(config.d_model, config.warmup_steps, 
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # Define loss function with label smoothing
    criterion = LabelSmoothingLoss(config.label_smoothing, vocab_size, ignore_index=config.pad_token_id)    
    # Train model
    if args.mode == 'train':
        train_model(model, train_loader, val_loader, optimizer, criterion, device, config)
    # Evaluate model
    elif args.mode == 'evaluate':
        evaluate_model(model, val_loader, criterion, device, config)
    # Test model
    elif args.mode == 'test':
        evaluate_model(model, test_loader, criterion, device, config, is_test=True)
    # Save model
    torch.save(model.state_dict(), config.model_save_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Model")
    parser.add_argument('--config_file', type=str, default='config/config.yaml', help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'test'], default='train', help='Mode to run the script in')
    args = parser.parse_args()
    main(args)