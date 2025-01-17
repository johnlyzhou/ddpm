import argparse

import torchvision
from torch.utils.data import DataLoader
import lightning as L

from dataset import FilteredMNIST
from ddpm import DDPM

config = {
    "num_channels": 1,
    "image_size": 28,
    "num_timesteps": 1000,
    "noise_schedule": "linear",
    "beta_min": 1e-4,
    "beta_max": 0.02,
    "learning_rate": 1e-3,
    "time_embed_dim": 256,
    "encoder_channels": (64, 128, 256),
    "decoder_channels": (256, 128, 64),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--max_epochs', type=int, default=100)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    args = parser.parse_args()
    config["batch_size"] = args.batch_size
    config["max_epochs"] = args.max_epochs

    dataset = FilteredMNIST(root="./data", label=6, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=args.num_workers)
    print(f"Number of samples: {len(dataset)}")
    
    model = DDPM(config)
    trainer = L.Trainer(max_epochs=config["max_epochs"])
    trainer.fit(model=model, train_dataloaders=train_loader)
