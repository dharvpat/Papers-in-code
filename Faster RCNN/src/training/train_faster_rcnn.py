import torch
from torch import optim
from models.faster_rcnn import get_faster_rcnn_model
from datasets.custom_dataset import create_dataloaders

def train_faster_rcnn(num_epochs=20, batch_size=8, learning_rate=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the dataloaders
    train_loader, _ = create_dataloaders(batch_size)
    
    # Initialize the model
    model = get_faster_rcnn_model(num_classes=21).to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    print("Training complete.")