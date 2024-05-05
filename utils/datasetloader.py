from os import listdir
from PIL import Image
from torch.utils.data import Dataset
import torch

from utils.common import transform_img, device


class DatasetLoader(Dataset):
    def __init__(self, num_classes, path, width=128, height=128):
        self.num_classes = num_classes
        self.width = width
        self.height = height

        self.path = path
        self.class_dirs = len(listdir(self.path))
        self.images = []
        self.targets = []

        self.load_images(aug=False)
        self.load_images(aug=True)

    def load_images(self, aug=False):
        for label in range(self.class_dirs):
            current_dir = f'{self.path}/{label}'
            img_paths = listdir(current_dir)
            for img in img_paths:
                image = Image.open(f'{current_dir}/{img}').convert('RGB')

                self.images.append(transform_img(image, self.width, self.height, aug).to(device))

                target = [0 for _ in range(self.num_classes)]
                target[int(label)] = 1
                self.targets.append(target)

    def __getitem__(self, item):
        image = self.images[item]
        target = self.targets[item]
        return {'image': image, 'target': target}

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    images = [item['image'] for item in batch]
    targets = [item['target'] for item in batch]
    images = torch.stack(images, dim=0)
    targets = torch.Tensor(targets).to(device)
    return images, targets
