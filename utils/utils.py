import numpy as np
from sklearn.decomposition import PCA, FastICA

import torch
from torch.utils.data import Dataset, DataLoader

def ICA(data):  # data shape [1, 1, length]
    # separate into three parts
    components = 3    # or 5
    data = np.array(data)
    ica = FastICA(n_components=components)
    return ica

def type_align(data):
    # align the data type into nparray or tensor

    return

if __name__ == "__main__":
    a = 1
