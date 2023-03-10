import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import tqdm
from utils.train_eval import train
from models.the_model import BSSmodel
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from datasets.datasets import perturbed_dataset, separated_dataset, perturbed_test_dataset, separated_test_dataset

def parse_args_and_config(**parser_kwargs):
    parser = argparse.ArgumentParser(description='ddd')
    parser.add_argument('--config', type=str, default="set_1.yaml") #, required=True
    parser.add_argument('--seed', type=int, default=9574, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp')
    parser.add_argument('--lambda', type=float, default=0.5, help='The restriction coefficient')
    parser.add_argument('--timesteps', type=int, default=500, help='Sample time steps')
    
    args = parser.parse_args()

    path = "configs/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    device = torch.device('cuda:3')
    new_config.device = device

    # no need seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    dataloader = DataLoader(separated_dataset(), batch_size=config.train.batch_size, shuffle=True)
    train(
        model=model,
        config=config,
        train_loader=dataloader,
    )
