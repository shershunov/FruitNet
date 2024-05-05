from sys import stdout
import torch
from torch.nn import functional as F
from tqdm import tqdm


def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    best_model = model
    best_train_loss = float('inf')
    dataloader_length = len(dataloader)
    losses_train = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0
        print('Epoch\tGPU_mem')
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
        tqdm_desc = f'{epoch + 1}/{num_epochs}    {mem}'

        with tqdm(total=dataloader_length, desc=tqdm_desc, file=stdout, leave=True) as pbar:
            for batch, (images, target) in enumerate(dataloader):
                optimizer.zero_grad()

                output = model(images)
                loss = F.binary_cross_entropy(output, target)
                epoch_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                pbar.update(1)

            pbar.close()
            epoch_train_loss /= dataloader_length
            losses_train.append(epoch_train_loss)
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_model = model.state_dict()
            print(f'\tLoss: {epoch_train_loss:.4f}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.save(best_model, 'best.pt')
    return losses_train
