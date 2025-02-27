import torch
import matplotlib.pyplot as plt
from diffusion_model import DiffusionModel
from encoder import Encoder
from decoder import Decoder
from data_loader import get_data_loader

if torch.cuda.is_available():
  device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

encoder = Encoder().to(device)
decoder = Decoder().to(device)
model = DiffusionModel(encoder, decoder).to(device)
model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
model.eval()

_, test_loader = get_data_loader(batch_size=1)

with torch.no_grad():
    for batch in test_loader:
        x, _ = batch
        x = x.to(device)
        predicted = model(x)
        x = x.squeeze(0).cpu().numpy()
        predicted = predicted.squeeze(0).cpu().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(x[0], cmap="gray")
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(predicted[0], cmap="gray")
        axs[1].set_title("Reconstructed Image")
        axs[1].axis("off")

        plt.show()
        break
