import torch
from sklearn.neighbors import KNeighborsClassifier
from datasets.custom_dataset import create_dataloaders
from models.contrastive_model import ContrastiveModel
from torchvision import models

def evaluate_contrastive_learning(batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the dataloaders
    _, test_loader = create_dataloaders(batch_size)
    
    # Load the trained model
    base_model = models.resnet18(pretrained=True)
    model = ContrastiveModel(base_model).to(device)
    model.load_state_dict(torch.load('contrastive_model.pth'))
    model.eval()
    
    embeddings_list, labels_list = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            embeddings = model(images).cpu()
            embeddings_list.append(embeddings)
            labels_list.append(labels)
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Evaluate using k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embeddings, labels)
    accuracy = knn.score(embeddings, labels)
    
    print(f"Evaluation Accuracy: {accuracy:.4f}")