import numpy as np

import ray
from ray import air, tune
from ray.air import session
from ray.tune.search.ax import AxSearch

# def hijack()

def objective(config):
    
    return 

if __name__ == "__main__":
    search_space = {
        "_lambda": tune.uniform(0.0, 1,0)
    }

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
        
        )
    )