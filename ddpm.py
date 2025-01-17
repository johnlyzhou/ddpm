import torch
import torch.nn.functional as F
import lightning as L
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from networks import UNet
from utils import make_broadcastable, linear_schedule, cosine_schedule, grayscale_to_pil


class SampleCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        sample_image = pl_module.sample(1).squeeze()
        grayscale_to_pil(sample_image).show()


class DDPM(L.LightningModule):
    """A simplified implementation of Denoising Diffusion Probabilistic Model (DDPM; Ho et al., 2022)."""
    def __init__(self):
        super().__init__()
        config = {
            "num_channels": 1,
            "image_size": 28,
            "num_timesteps": 25,
            "noise_schedule": "linear",
            "beta_min": 1e-4,
            "beta_max": 0.02
        }
        self.num_channels = config["num_channels"]
        self.net = UNet(self.num_channels)
        self.image_size = config["image_size"]
        self.num_timesteps = config["num_timesteps"]

        if config["noise_schedule"] == 'linear':
            self.register_buffer("betas", linear_schedule(config["beta_min"], config["beta_max"], self.num_timesteps))
        elif config["noise_schedule"] == 'cosine':
            self.register_buffer("betas", cosine_schedule(config["beta_min"], config["beta_max"], self.num_timesteps))

        # Precompute some fixed values
        self.register_buffer("alphas", 1 - self.betas)
        # For training: see Algorithm 1
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))
        # For sampling: see Algorithm 2
        self.register_buffer("reciprocal_sqrt_alphas", 1.0 / self.alphas)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # See Section 3.2
        self.register_buffer("posterior_variance", self.betas * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.net.parameters(), lr=1e-4)

    @torch.no_grad()
    def diffuse(self, x_0, t):
        """
        Add noise, appropriately scaled according to t, to a given image x_0.
        :param x_0: Batch of images of shape (bsz, num_channels, height, width).
        :param t: Diffusion timestep (bsz,).
        :return: Image with noise added according to timestep t and ground truth noise.
        """
        noise = torch.randn_like(x_0).to(self.device)
        mean_coefficient = make_broadcastable(self.sqrt_alphas_cumprod[t], x_0.shape)
        var_coefficient = make_broadcastable(self.sqrt_one_minus_alphas_cumprod[t], x_0.shape)
        x_t = mean_coefficient * x_0 + var_coefficient * noise
        return x_t, noise

    def training_step(self, x_0):
        """
        Training step according to Algorithm 1. Sample noise and diffusion time step t, scale
          noise and add to the images, then predict the added noise given the noisy image and t.
        :param x_0: Batch of images of shape (bsz, num_channels, height, width).
        """
        bsz = x_0.size(0)
        t = torch.randint(self.num_timesteps, (bsz,)).to(self.device)
        x_t, noise = self.diffuse(x_0, t)
        noise_hat = self.net(x_t, t)
        loss = F.l1_loss(noise_hat, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        sample_image = self.sample(1).squeeze()
        grayscale_to_pil(sample_image).show()

    @torch.no_grad()
    def reverse_step(self, x_t, t):
        """
        Sample p(x_{t - 1} | x_t, t) according to Algorithm 2 of Ho et al. (2022). Note that we use 0-indexing for
          convenience, making x_{-1} the final denoised image rather than x_0.
        :return: Sampled image x_{t - 1}.
        """
        noise_hat = self.net(x_t, t) * make_broadcastable(self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t],
                                                          x_t.shape)
        posterior_mean = (x_t - noise_hat) * make_broadcastable(self.reciprocal_sqrt_alphas[t], x_t.shape)

        if torch.any(t == 0):
            return posterior_mean
        else:
            posterior_noise = (make_broadcastable(self.posterior_variance[t], x_t.shape) *
                               torch.randn_like(x_t).to(self.device))

            return posterior_mean + posterior_noise

    @torch.no_grad()
    def sample(self, bsz):
        """
        Sample a batch of random noise and run the reverse diffusion process to sample images from the model.
        """
        x_t = torch.randn((bsz, self.num_channels, self.image_size, self.image_size)).to(self.device)
        for t in reversed(range(self.num_timesteps)):
            t = torch.ones((bsz,), dtype=torch.int).to(self.device) * t
            x_t = self.reverse_step(x_t, t)
            x_t = torch.clamp(x_t, -1.0, 1.0)  # Clamp to valid pixel values
        return x_t
