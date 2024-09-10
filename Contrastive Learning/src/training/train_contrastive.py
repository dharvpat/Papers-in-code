import torch
from torch import optim
from models.contrastive_model import ContrastiveModel
from algorithms.contrastive_learning import SupervisedContrastiveLoss
from datasets.custom_dataset import create_dataloaders
from torchvision import models

def train_contrastive_learning(num_epochs=100, batch_size=128, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the dataloaders
    train_loader, _ = create_dataloaders(batch_size)
    
    # Initialize the model
    base_model = models.resnet18(pretrained=True)
    model = ContrastiveModel(base_model).to(device)
    
    # Loss function and optimizer
    criterion = SupervisedContrastiveLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    print("Training complete.")