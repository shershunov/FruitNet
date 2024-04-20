from os import listdir
from random import shuffle
from PIL import Image
from torch.utils.data import Dataset
import torch

from utils.common import transform_img, device


class DatasetLoader(Dataset):
    def __init__(self):
        self.path = 'data/train'
        self.paths = len(listdir(self.path))
        self.images = []
        self.targets = []

        self.elements = 0

        self.load_images(aug=False)
        self.load_images(aug=True)

    def load_images(self, aug=False):
        for label in range(self.paths):
            img_paths = listdir(f'{self.path}/{label}')
            shuffle(img_paths)
            for img in img_paths:
                self.elements += 1
                full_path = f'{self.path}/{label}/{img}'
                image = Image.open(full_path).convert('RGB')

                self.images.append(transform_img(image, aug).to(device))

                target = [0, 0, 0]
                target[int(label)] = 1
                self.targets.append(target)

    def __getitem__(self, item):
        image = self.images[item]
        target = self.targets[item]
        return {'image': image, 'target': target}

    def __len__(self):
        return self.elements


def collate_fn(batch):
    images = [item['image'] for item in batch]
    targets = [item['target'] for item in batch]
    images = torch.stack(images, dim=0)
    targets = torch.Tensor(targets).to(device)
    return images, targets
