
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.autoencoder import Autoencoder

# 1. Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Load CIFAR-10
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# 3. Initialize autoencoder
model = Autoencoder(latent_channels=8).to(device)

# 4. Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train autoencoder
epochs = 10

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, _ in train_loader:

        images = images.to(device)

        # Reconstruct images
        reconstructed = model(images)

        # Compare reconstructed images with originals
        loss = criterion(reconstructed, images)

        # Update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.6f}"
    )

# 6. Save trained model
torch.save(
    model.state_dict(),
    "autoencoder.pth"
)

print("Autoencoder training complete!")
print("Model saved as autoencoder.pth")
