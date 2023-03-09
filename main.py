# main function

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import yaml
import sys
import os
import torch.utils.tensorboard as tb
from torch.utils.data import dataset

from models.the_model import BSSmodel
from datasets.datasets import perturbed_test_dataset, separated_test_dataset

import time

now = time.localtime()
now_time = time.strftime("%m-%d_%H:%M", now)

# args: alternative arguments
# config: parameters have been setted up already
def parse_args_and_config(**parser_kwargs):
    parser = argparse.ArgumentParser(description='ddd')
    parser.add_argument('--config', type=str, default="set_1.yaml") #, required=True
    parser.add_argument('--seed', type=int, default=9574, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp')
    parser.add_argument('--lambda', type=float, default=0.5, help='The restriction coefficient')
    parser.add_argument('--timesteps', type=int, default=500, help='Sample time steps')
    
    args = parser.parse_args()

    path = "config/" + args.config
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

def get_data(number=10):
    dataset = perturbed_test_dataset()
    ground_truth = separated_test_dataset()
    data = dataset.data
    ground_truth_data = ground_truth.data
    pick = np.random.choice(data.shape[0], size=number, replace=False)
    pick_test_data = data[pick]
    pick_ground_truth = ground_truth_data[pick]
    return pick_test_data, pick_ground_truth

def BSS():
    args, config = parse_args_and_config()
    # get args and config
    # args can change in the main, config is defined in yaml

    perturbed_data, ground_truth = get_data()

    try:
        model = BSSmodel(args, config)
        result = model.hijack_sample(perturbed_data)
    except Exception:
        logging.error()
    
    now = time.localtime()
    now_time = time.strftime("%m-%d_%H:%M.npz", now)
    
    save_path = config.path.save_path
    file_path = save_path + now_time
    np.savez(file_path, result=result, ground_truth=ground_truth)

    return 0


if __name__ == "__main__":
    # get data (seperate and perturbed)
    BSS()
