import numpy as np
import yaml
from sklearn.decomposition import PCA, FastICA
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np

def ICA(data):  # data shape [1, 1, length]
    # separate into three parts
    components = 3    # or 5
    data = np.array(data)
    ica = FastICA(n_components=components)
    return ica  # []

def type_align(data):
    # align the data type to torch tensor
    if isinstance(data, torch.Tensor):
        return data.type(torch.float32)
    elif isinstance(data, np.array):
        return torch.from_numpy(data).type(torch.float32)
