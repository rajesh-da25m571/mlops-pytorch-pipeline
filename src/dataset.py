# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(dataset_name: str, train: bool):
    dataset_name = dataset_name.lower()
    
    if dataset_name == "fashion_mnist":
        # Pad MNIST from 28x28 to 32x32 so it shares sizing parameters cleanly
        return transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530])
        ])
        
    elif dataset_name == "cifar10":
        t_list = []
        if train:
            t_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        return transforms.Compose(t_list)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_dataloaders(dataset_name: str, data_dir: str, batch_size: int = 64, num_workers: int = 2):
    name = dataset_name.lower()
    
    if name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(name, True))
        val_ds  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(name, False))
    elif name == "fashion_mnist":
        train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=get_transforms(name, True))
        val_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=get_transforms(name, False))
    else:
        raise ValueError(f"Unsupported dataset target: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
