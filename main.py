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


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(description='ddd')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=9574, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp')
    return

def main():
    args, config = get_parser()


    try:
        runner = 
        runner
    return

if __name__ == "__main__":
    
    run()
