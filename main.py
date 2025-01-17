import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import lightning as L

from diffusion import DDPM, SampleCallback


class FilteredMNIST(Dataset):
    def __init__(self, root, label, train=True, transform=None, download=True):
        """
        Initialize the dataset by filtering for the specified label.

        Args:
            root (str): Path to save/load the MNIST dataset.
            label (int): The label to filter (0-9).
            train (bool): Whether to use the training set. Defaults to True.
            transform (callable, optional): A function/transform to apply to the images.
            download (bool): Whether to download the dataset if it does not exist. Defaults to True.
        """
        self.label = label
        self.transform = transform

        # Load the MNIST dataset
        self.mnist = torchvision.datasets.MNIST(root=root, train=train, download=download)

        # Filter indices where the target matches the specified label
        self.indices = [i for i, target in enumerate(self.mnist.targets) if target == label]

    def __len__(self):
        """Return the number of samples in the filtered dataset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Retrieve the sample at the specified index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the actual index in the original MNIST dataset
        actual_idx = self.indices[idx]
        image, target = self.mnist[actual_idx]

        if self.transform:
            image = self.transform(image)
            image = (image - 0.5) * 2  # Normalize to [-1, 1]

        return image


if __name__ == "__main__":
    # Create a FilteredMNIST dataset for the digit 7
    dataset = FilteredMNIST(root="./data", label=7, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Print the number of samples in the dataset
    print(f"Number of samples: {len(dataset)}")

    config = {
        "num_channels": 1,
        "image_size": 28,
        "num_timesteps": 25,
        "noise_schedule": "linear",
        "beta_min": 1e-4,
        "beta_max": 0.02
    }

    model = DDPM(config)
    trainer = L.Trainer(max_epochs=100, callbacks=[SampleCallback()])
    trainer.fit(model=model, train_dataloaders=train_loader)
