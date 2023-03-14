import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

def train(
    model,
    config,
    train_loader,
):
    batch_size = config.batch_size
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=1e-6, eps=1e-5)
    foldername = config.save_path
    output_path = foldername + "/model.pth"
    tqdm_epoch = tqdm.trange(config.epoch)
    for epoch_no in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in train_loader:  # why start = 1
            optimizer.zero_grad()
            loss = model(train_loader)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item() * x.shape
            num_items += x.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        torch.save(model.state_dict(), output_path)

    return 0

def evaluate():
    return 0