from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt


def build_transform(is_train):
    transform = []

    if is_train:
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.RandomResizedCrop(224))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        transforms.Resize((224, 224))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform)

def get_dataloader(data_dir, batch_size, is_train):
    transform = build_transform(is_train)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if is_train:
        loader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=4, shuffle=True, pin_memory=True)
        print("Train loader is prepared.")
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=4, shuffle=False, pin_memory=True)
        print("Test loader is prepared.")

    return loader

if __name__ == '__main__':
    pass