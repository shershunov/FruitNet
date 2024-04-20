from sys import stdout
import torch
from torch.nn import functional as F
from tqdm import tqdm


def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    best_model_wts = model
    best_val_loss = float('inf')
    dataloader_len = len(dataloader)
    losses_train = []

    for epoch in range(num_epochs):
        train_loss = 0
        print('Epoch\tGPU_mem')
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
        tqdm_desc = f'{epoch + 1}/{num_epochs}\t\t{mem}'

        with tqdm(total=dataloader_len, desc=tqdm_desc, file=stdout, leave=True) as pbar:
            for batch, (images, target) in enumerate(dataloader):
                optimizer.zero_grad()

                output = model(images)
                loss = F.binary_cross_entropy(output, target)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
                pbar.update(1)

            pbar.close()
            train_loss /= dataloader_len
            losses_train.append(train_loss)
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_model_wts = model.state_dict()
            print(f'\tLoss: {train_loss:.4f}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.save(best_model_wts, 'best.pt')
    return losses_train
