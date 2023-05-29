from typing import Any
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt


def build_transform(is_train):
    transform = []

    if is_train:
        transform.append(transforms.RandomResizedCrop(224))
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        transform.append(transforms.Resize((224, 224)))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform)

class StairDataset(Dataset):
    def __init__(self, json_file, root, transform=None):
        self.img_path = []
        self.labels = []
        self.transfrom = transform
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
            for data in raw_data:
                self.img_path.append(os.path.join(root, data['dir']))
                self.labels.append(data['label'])

    def __getitem__(self, index: Any) -> Any:
        path = self.img_path[index]
        label = self.labels[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transfrom is not None:
            img = self.transfrom(img)
        return img, label
    
    def __len__(self):
        return len(self.labels)


def get_dataloader(data_dir, json_file, batch_size, is_train):
    transform = build_transform(is_train)
    dataset = StairDataset(json_file, data_dir, transform)
    if is_train:
        loader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=4, shuffle=True)
        print("Train loader is prepared.")
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=4, shuffle=False)
        print("Test loader is prepared.")

    return loader

if __name__ == '__main__':
    dataset = StairDataset("public_valid.json", "../../dataset/public", build_transform(False))

    for img, label in dataset:
        print(label, img.shape)
        break

    loader = get_dataloader("../../dataset/public", "public_valid.json", 8, False)

    for (imgs, labels) in loader:
        print(labels, imgs.shape)
        break