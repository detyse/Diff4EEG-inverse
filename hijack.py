import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import os
from torch.utils.data import DataLoader
from datasets.datasets import hijack_dataset
from utils.train_eval import hijack
from models.the_model import BSSmodel


def parse_args_and_config(**parser_kwargs):
    parser = argparse.ArgumentParser(description='ddd')
    parser.add_argument('--config', type=str, default="set_1.yaml") #, required=True
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--timesteps', type=int, default=500, help='Sample time steps')
    parser.add_argument('--ckpt', type=str, default='model', help="ckpt to load model")
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

if __name__ == '__main__':
    args, config = parse_args_and_config()
    
    dataloader = DataLoader(hijack_dataset(), batch_size=args.batch_size, shuffle=True)
    
    model = BSSmodel(args, config)
    try:
        ckpt = os.path.join(config.hijack.model_path, args.ckpt)
        model.load_state_dict(torch.load(ckpt))
        print("model loaded")
    except:
        print("ckpt does not exist")

    hijack(model=model, dataloader=dataloader, args=args, config=config)
