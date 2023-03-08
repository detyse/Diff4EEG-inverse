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

# args: alternative arguments
# config: parameters have been setted up already
def parse_args_and_config(**parser_kwargs):
    parser = argparse.ArgumentParser(description='ddd')
    parser.add_argument('--config', type=str, default="set_1.yaml") #, required=True
    parser.add_argument('--seed', type=int, default=9574, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp')
    args = parser.parse_args()

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

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

def main():
    args, config = parse_args_and_config()

    try:
        a = 1
    except Exception:
        logging.error()
    
    return 0

if __name__ == "__main__":
    main()
