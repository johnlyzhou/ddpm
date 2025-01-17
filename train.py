import torchvision
from torch.utils.data import DataLoader
import lightning as L

from dataset import FilteredMNIST
from ddpm import DDPM, SampleCallback

if __name__ == "__main__":
    # Create a FilteredMNIST dataset for the digit 7
    dataset = FilteredMNIST(root="./data", label=7, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Print the number of samples in the dataset
    print(f"Number of samples: {len(dataset)}")
    
    model = DDPM()
    trainer = L.Trainer(max_epochs=100, callbacks=[SampleCallback()])
    trainer.fit(model=model, train_dataloaders=train_loader)
