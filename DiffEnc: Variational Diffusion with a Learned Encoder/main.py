import torch
from diffusion_model import DiffusionModel
from encoder import Encoder
from decoder import Decoder
from data_loader import get_data_loader
from torch.optim import Adam
from utils import save_checkpoint, log_metrics

epochs = 100
learning_rate = 1e-2
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

train_loader, test_loader = get_data_loader(batch_size=32)

encoder = Encoder().to(device)
decoder = Decoder().to(device)
model = DiffusionModel(encoder, decoder, device).to(device)

# Optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training Loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()  # Update learning rate

    log_metrics(epoch, epoch_loss)
    save_checkpoint(model, optimizer, epoch, epoch_loss)
    
    print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")