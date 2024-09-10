import torch
from model.transformer import Transformer
from utils.data_loader import get_data_loaders
from config.config import Config

def infer():
    # Load configuration
    cfg = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = Transformer(d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers, d_ff=cfg.d_ff, dropout=cfg.dropout)
    model.load_state_dict(torch.load(cfg.model_save_path))
    model.eval()
    model.to(device)

    # Load data
    tokenizer = ...  # Load or initialize your tokenizer here
    input_data = ...  # Load or prepare your input data here
    data_loader = get_data_loaders(input_data, None, tokenizer, batch_size=cfg.batch_size, shuffle=False)

    # Run inference
    with torch.no_grad():
        for input_seq in data_loader:
            input_seq = input_seq.to(device)
            outputs = model(input_seq)
            preds = outputs.argmax(dim=-1)
            # Process and output predictions
            print(preds)

if __name__ == "__main__":
    infer()