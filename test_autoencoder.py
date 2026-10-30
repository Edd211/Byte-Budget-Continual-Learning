import torch
from models.autoencoder import Autoencoder

model = Autoencoder(latent_channels=8)

image = torch.rand(1, 3, 32, 32)

compressed = model.encoder(image)
reconstructed = model.decoder(compressed)

print("Original shape:", image.shape)
print("Compressed shape:", compressed.shape)
print("Reconstructed shape:", reconstructed.shape)

print("Original bytes:", image.nelement() * image.element_size())
print("Compressed bytes:", compressed.nelement() * compressed.element_size())