import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import save_in_time, save_in_time_hijack, evaluate
from models.the_model import BSSmodel
import os
import functools

def train(
    model,
    args,
    config,
    train_loader,
):
  
    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=1e-6, eps=1e-5)
    
    foldername = config.train.save_path
    output_path = os.path.join(foldername, args.model_save)
    tqdm_epoch = tqdm(range(config.train.epoch))
    for epoch_no in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in train_loader:  # why start = 1
            x = x.squeeze(1)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        torch.save(model.state_dict(), output_path)

    return 0


def sample(model,
           args,
           config,):
    ckpt = os.path.join(config.sample.model_path, args.ckpt)
    model.load_state_dict(torch.load(ckpt))
    data_shape = config.sample.data_shape
    result = model.sample(data_shape)
    result = result.cpu().numpy()
    save_path = config.sample.result_path
    info = 'sample' + 'datalength:' + str(config.sample.data_shape[0]) + args.save_info
    save_in_time(result, save_path, info)
    return 0


def hijack(model, 
           args,
           config,
           hijack_loader,
           _lambda,
           ):
    num_items = 0
    error = 0.

    for perturbed_data, ground_truth in hijack_loader:
        
        result = model.hijack_sample(perturbed_data, _lambda)

        result = result.cpu().numpy()
        ground_truth = ground_truth.numpy()
        save_path = config.hijack.result_path
        info = 'hijack' + 'datalength:' +  str(config.hijack.data_shape[0])
        save_in_time_hijack(result, ground_truth, save_path, info)
        
        num_items += perturbed_data.shape[0]
        error += evaluate(result, ground_truth)
    mean_err = error / num_items
    
    return mean_err


def hijack_optim(hijack_fn, _lambda, best_lambda, best_err):
    err = hijack_fn(_lambda)
    if err < best_err:
        best_err = err
        best_lambda
    return best_lambda, best_err