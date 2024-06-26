from os import listdir
from torchvision.transforms import v2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_img(img, width, height, aug=False):
    transform_norm = v2.Compose([
        v2.Resize(size=(width, height)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_aug = v2.Compose([
        v2.Resize(size=(width, height)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(20),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform_aug(img) if aug else transform_norm(img)


def get_num_classes(path):
    return len(listdir(path))
