
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Convert images into PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download CIFAR-10 testing dataset
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Inspect dataset
print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Image data type:", images.dtype)
print("Image size in bytes:", images[0].nelement() * images[0].element_size())
