from os import listdir
from PIL import Image
from numpy import asarray
import torch
import matplotlib.pyplot as plt

from utils.common import transform_img, device


def predict(model, image):
    image = transform_img(image, aug=False).to(device)
    return model(image.unsqueeze(0))


def draw_graph(losses):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show()


def test_val(model):
    model.load_state_dict(torch.load('best.pt'))
    model.eval()
    path = 'data/val'
    val_photos = listdir(path)
    classes = {0: 'apple', 1: 'banana', 2: 'kiwi'}

    for photo in val_photos:
        img_path = f'{path}/{photo}'

        img = Image.open(img_path).convert('RGB').resize((256, 256))
        predictions = predict(model, img).tolist()[0]

        img = asarray(img)
        plt.imshow(img)

        plt.text(0, -10,
                 f'{list(map(lambda x: round(x, 4), predictions))} {classes[predictions.index(max(predictions))]}',
                 color='black', fontsize=14)

        plt.axis('off')
        plt.show()
