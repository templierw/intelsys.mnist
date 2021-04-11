from torchvision import datasets as dt, transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader


def loadMNISTDatasets(path='./dataset'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]) # values taken from https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/6

    dataset = dt.MNIST(root=path, train=True, download=True, transform=transform)

    val_split = int(len(dataset) * 0.2)
    train_split = len(dataset) - val_split
    train_dataset, val_dataset = random_split(dataset, [train_split, val_split])

    holdback_dataset = dt.MNIST(root=path, train=False, download=True, transform=transform)

    return train_dataset, val_dataset, holdback_dataset


def getMNISTLoaders(datasets, batch_size=4, num_workers=2):
    train_dataset, test_dataset, val_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, val_loader
