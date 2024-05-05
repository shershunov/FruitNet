from os import listdir
from PIL import Image
from numpy import asarray
import torch
import matplotlib.pyplot as plt

from utils.common import transform_img, device


def predict(model, image, width, height):
    image = transform_img(image, width, height, aug=False).to(device)
    return model(image.unsqueeze(0))


def draw_graph(losses):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show()


def test_val(model, path, width=128, height=128):
    model.load_state_dict(torch.load('best.pt'))
    model.eval()

    val_photos = listdir(path)
    classes = {0: 'apple', 1: 'banana', 2: 'kiwi'}

    for photo in val_photos:
        img = Image.open(f'{path}/{photo}').convert('RGB')
        predictions = predict(model, img, width, height).tolist()[0]

        plt.imshow(asarray(img))

        plt.text(0, -10,
                 f'{list(map(lambda x: round(x, 4), predictions))} {classes[predictions.index(max(predictions))]}',
                 color='black', fontsize=14)

        plt.axis('off')
        plt.show()
