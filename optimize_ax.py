import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

import yaml
import functools

from utils.train_eval import hijack, hijack_optim
from utils.utils import random_select
from models.the_model import BSSmodel
from datasets.datasets import hijack_dataset

# def hijack()
  
def parse_args_and_config(**parser_kwargs):
    parser = argparse.ArgumentParser(description='ddd')
    parser.add_argument('--config', type=str, default="set_1.yaml") #, required=True
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--timesteps', type=int, default=500, help='Sample time steps')
    parser.add_argument('--ckpt', type=str, default='model.pth', help="ckpt to load model")
    parser.add_argument('--batch_size', type=int, default=32, help='')
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

if __name__ is "__main__":
    args, config = parse_args_and_config()

    seach_space = {
        '_lambda':[i * 0.01 for i in range(100)]
    }

    search_times = 50
    best_lambda = None
    best_err = None
    
    model = 1
    hijack_fn = functools.partial(hijack, model=model, args=args, config=config)

    for _ in range(search_times):
        _lambda = random_select(seach_space)
        best_lambda, best_err = hijack_optim(hijack_fn, _lambda, best_lambda, best_err)

    print("best_lambda is: ", best_lambda)
    # print("best_err is: ", best_err)
    