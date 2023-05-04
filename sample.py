import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import os
from models.the_model import BSSmodel
from utils.train_eval import sample

def get_args_and_config():
    parser = argparse.ArgumentParser(description='Sample EEG')
    parser.add_argument('--config', type=str, default='set_1.yaml', help='')
    parser.add_argument('--timesteps', type=int, default=300, help='')
    parser.add_argument('--ckpt', type=str, default="sssd_2.pth", help='')
    parser.add_argument('--save_info', type=str, default='', help='')
    args = parser.parse_args()

    path = "configs/" + args.config
    with open(path, "r") as f:
        config = yaml.unsafe_load(f)
    new_config = dict2namespace(config)
    
    device = torch.device('cuda:3')
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
    args, config = get_args_and_config()
    device = torch.device("cuda:3")
    model = BSSmodel(args, config)
    
    try:
        ckpt = os.path.join(config.hijack.model_path, args.ckpt)
        print(ckpt)
        model.load_state_dict(torch.load(ckpt))
        print("::: model loaded :::")
    except:
        print("::: ckpt does not exist :::")
    
    model.eval()
    with torch.no_grad():
        sample(model=model, args=args, config=config)