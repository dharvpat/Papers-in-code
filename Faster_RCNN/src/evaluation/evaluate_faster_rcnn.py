import torch
from datasets.custom_dataset import create_dataloaders
from models.faster_rcnn import get_faster_rcnn_model

def evaluate_faster_rcnn(batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the dataloaders
    _, test_loader = create_dataloaders(batch_size)
    
    # Load the trained model
    model = get_faster_rcnn_model(num_classes=21).to(device)
    model.load_state_dict(torch.load('faster_rcnn_model.pth'))
    model.eval()
    
    # Evaluation logic (e.g., mAP calculation) goes here
    # Placeholder for evaluating model performance on the test set
    print("Evaluation complete.")