
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.autoencoder import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Load trained autoencoder
model = Autoencoder(latent_channels=8).to(device)
model.load_state_dict(
    torch.load("autoencoder.pth", map_location=device, weights_only=True)
)

model.eval()

# Evaluate reconstruction quality
total_loss = 0
criterion = torch.nn.MSELoss(reduction="sum")

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        reconstructed = model(images)

        total_loss += criterion(reconstructed, images).item()

test_loss = total_loss / (len(test_dataset) * 3 * 32 * 32)

print("Test reconstruction MSE:", test_loss)

# Get images for visualization
images, _ = next(iter(test_loader))
images = images[:8].to(device)

with torch.no_grad():
    reconstructed = model(images)

# Display original and reconstructed images
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

for i in range(8):
    original = images[i].cpu().permute(1, 2, 0)
    decoded = reconstructed[i].cpu().permute(1, 2, 0)

    axes[0, i].imshow(original)
    axes[0, i].axis("off")
    axes[0, i].set_title("Original")

    axes[1, i].imshow(decoded)
    axes[1, i].axis("off")
    axes[1, i].set_title("Reconstructed")

plt.tight_layout()
plt.savefig("reconstruction_results.png")
plt.show()
