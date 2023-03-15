import torch
import torch.nn as nn
import numpy as np

from datasets.datasets import perturbed_test_dataset, separated_test_dataset

from ray import air, tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch

from models.the_model import BBSmodel

import argparse

from utils.train_eval import hijack

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="set_1.yaml")

    return 0

def objecive(config):

    return 0

def lambda_function(_lambda):
    
    return 

def get_bias():

    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument()

    print("Best lambda found were: ", results.get_best_result().config)