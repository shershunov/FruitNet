import torch
from torch.utils.data import DataLoader

from models.tfn import TFN
from utils.datasetloader import DatasetLoader, collate_fn
from utils.train import train_model
from utils.test import test_val, draw_graph
from utils.common import device

BATCH = 120
EPOCHS = 15

model = TFN().to(device)

dataset = DatasetLoader()
dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

losses_train = train_model(model, dataloader, optimizer, EPOCHS)

draw_graph(losses_train)
test_val(model)
