import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
from utils.train_eval import train, sample, hijack
from models.the_model import BSSmodel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets.datasets import perturbed_dataset, separated_dataset, hijack_dataset

def parse_args_and_config(**parser_kwargs):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default="set_1.yaml") #, required=True
    parser.add_argument('--timesteps', type=int, default=500, help='Sample time steps')
    parser.add_argument('--model_save', type=str, default="diffwave_4.pth", help='Save model name')
    parser.add_argument('--ckpt', type=str, default="diffwave_4.pth", help='Load model name')
    args = parser.parse_args()

    path = "configs/" + args.config
    with open(path, "r") as f:
        config = yaml.unsafe_load(f)
    new_config = dict2namespace(config)
    
    device = torch.device('cuda:2')
    new_config.device = device

    # no need seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    # torch.backends.cudnn.benchmark=True

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    args, config = parse_args_and_config()
    device = torch.device("cuda:3")
    model = BSSmodel(args, config)
    ckpt = os.path.join(config.hijack.model_path, args.ckpt)

    try:
        ckpt = os.path.join(config.hijack.model_path, args.ckpt)
        print(ckpt)
        model.load_state_dict(torch.load(ckpt))
        print("::: model loaded :::")
    except:
        print("::: ckpt does not exist :::")

    model.train()
    dataloader = DataLoader(separated_dataset(), batch_size=config.train.batch_size, shuffle=True)
    train(
        model=model,
        args=args,
        config=config,
        train_loader=dataloader,
    )
