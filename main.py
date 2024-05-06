import torch
from torch.utils.data import DataLoader

from models.tfn import TFN
from utils.datasetloader import DatasetLoader, collate_fn
from utils.train import train_model
from utils.test import test_val, draw_losses_graph
from utils.common import device, get_num_classes

train_path = 'data/train'
val_path = 'data/val'
BATCH = 60
EPOCHS = 128
WIDTH = 128
HEIGHT = 128
num_classes = get_num_classes(train_path)
learning_rate = 1e-2

model = TFN(num_classes).to(device)

dataset = DatasetLoader(num_classes, train_path, WIDTH, HEIGHT)
dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses_train = train_model(model, dataloader, optimizer, EPOCHS)

draw_losses_graph(losses_train)

test_val(model, val_path, WIDTH, HEIGHT)
