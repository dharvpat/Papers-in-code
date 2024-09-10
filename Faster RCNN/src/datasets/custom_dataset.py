import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection

def create_dataloaders(batch_size, dataset='VOC', num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset == 'VOC':
        train_dataset = VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)
        test_dataset = VOCDetection(root='./data', year='2012', image_set='val', download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader