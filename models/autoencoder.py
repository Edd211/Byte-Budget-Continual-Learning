
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_channels=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, 64,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_channels=8):
        super().__init__()

        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)

        return reconstructed
