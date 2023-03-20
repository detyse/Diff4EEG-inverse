# hyperparameter optimize

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from datasets.datasets import perturbed_dataset, separated_dataset
from datasets.datasets import hijack_dataset

from ray import air, tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch

from models.the_model import BSSmodel
import yaml
import argparse

from utils.train_eval import hijack, evaluate


def get_parser():
    parser = argparse.ArgumentParser(description='Optimization arguments set up')
    parser.add_argument('--config', type=str, default="set_1.yaml")
    parser.add_argument('--seed', type=int, default=9574, help='Random seed')
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


# Define an objective function
def objective(config):
    # load hijack
    dataset =  hijack_dataset()
    dataloader = DataLoader(dataset, )
    acc = hijack(config["_lambda"])
    return {"acc": acc}


if __name__ == "__main__":
    args, config = get_parser()

    model = BSSmodel(args, config)
    dataloader = DataLoader(hijack_dataset(), batch_size=config.optim.batch_size)

    # Define a search space
    search_space = {
        "_lambda": tune.grid_search(),
    }
    algo = OptunaSearch()



    # Start a Tune run and print the best result
    tuner = tune.Tuner(objective, param_space=search_space)
    results = tuner.fit
    print("Best lambda found were: ", results)