from collections import deque

import torch
import torch.nn as nn

from utils import make_broadcastable


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding to encode diffusion time steps."""
    def __init__(self, dim: int):
        super().__init__()
        assert(dim % 2 == 0)
        self.dim = dim

    def forward(self, time):
        """
        :param time: Shape (bsz,).
        :return: Positional embedding of shape (bsz, dim).
        """
        freqs = torch.pow(10000, torch.arange(0, self.dim, 2).float() / self.dim).to(time.device)  # (dim // 2,)
        embeddings = torch.zeros(time.size(0), self.dim).to(time.device)
        embeddings[:, 0::2] = torch.sin(time.unsqueeze(1) / freqs)  # (bsz, dim // 2)
        embeddings[:, 1::2] = torch.cos(time.unsqueeze(1) / freqs)
        return embeddings


class EncoderBlock(nn.Module):
    """
    U-Net encoder block modified for diffusion, which adds a learned positional embedding to the output of the first
      conv layer and downsamples with max pooling.
    """
    def __init__(self, in_chans, out_chans, time_embed_dim, kernel_size: int = 3, stride: int = 1, padding: int = 'same'):
        super(EncoderBlock, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_embed_dim, out_chans),
        )
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_chans),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_chans),
            nn.MaxPool2d(2, stride=2),
        )

    def forward(self, x, t):
        x = self.conv_layer_1(x)
        t = self.time_mlp(t)
        x = x + make_broadcastable(t, x.shape)
        return self.conv_layer_2(x)


class DecoderBlock(nn.Module):
    """
    U-Net decoder block modified for diffusion, which adds a learned positional embedding to the output of the first
      conv layer and upsamples with transposed convolutions.
    """
    def __init__(self, in_chans, out_chans, time_embed_dim, kernel_size: int = 3, stride: int = 1, padding: int = 'same'):
        super(DecoderBlock, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_embed_dim, out_chans),
        )
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_chans),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_chans),
            nn.ConvTranspose2d(out_chans, out_chans, kernel_size=2, stride=2),
        )

    def forward(self, x, t):
        x = self.conv_layer_1(x)
        t = self.time_mlp(t)
        x = x + make_broadcastable(t, x.shape)
        return self.conv_layer_2(x)


class UNet(nn.Module):
    """
    A simplified U-Net which additionally takes in a diffusion timestep t as input to every encoder/decoder block.
    """
    def __init__(self, in_chans: int):
        super(UNet, self).__init__()
        self.in_channels = in_chans
        self.encoder_channels = (64, 128, 256)
        self.decoder_channels = (256, 128, 64)
        self.time_embed_dim = 32


        self.time_embedding = nn.Sequential(
            SinusoidalPositionalEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.ReLU()
        )

        self.input_layer = nn.Conv2d(self.in_channels, self.encoder_channels[0], kernel_size=3, padding=1)
        self.encoders = nn.ModuleList([EncoderBlock(self.encoder_channels[i], self.encoder_channels[i + 1],
                                                    self.time_embed_dim) for i in range(len(self.encoder_channels) - 1)])
        # Multiply number of decoder channels by 2 to account for residual inputs
        self.decoders = nn.ModuleList([DecoderBlock(self.decoder_channels[i] * 2, self.decoder_channels[i + 1],
                                                    self.time_embed_dim) for i in range(len(self.decoder_channels) - 1)])
        self.output_layer = nn.Conv2d(self.decoder_channels[-1], self.in_channels, kernel_size=1)

    def forward(self, x, t):
        """
        U-Net forward pass with additional positional embedding and residual connections.
        :param x: Sample data with added noise
        :param t: Corresponding time step of the added noise
        """
        t = self.time_embedding(t)
        x = self.input_layer(x)

        residual_inputs = deque()
        for encoder_block in self.encoders:
            x = encoder_block(x, t)
            residual_inputs.append(x)

        for decoder_block in self.decoders:
            # Concatenate residual input in the channel dimension
            x = torch.cat([x, residual_inputs.pop()], dim=1)
            x = decoder_block(x, t)

        return self.output_layer(x)


if __name__ == '__main__':
    # Test UNet
    bsz = 2
    num_chans = 1
    max_diffusion_timesteps = 100
    net = UNet(num_chans)
    test_x = torch.randn(bsz, num_chans, 28, 28)
    test_t = torch.randint(max_diffusion_timesteps, (bsz,))
    y = net(test_x, test_t)
    print(y.shape)

