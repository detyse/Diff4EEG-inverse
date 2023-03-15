import numpy as np
import yaml
from sklearn.decomposition import PCA, FastICA
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np

import os
import shutil
from datetime import datetime

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

def save_in_time(result, save_path, *args):
    folder_name = 'output_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = os.path.join(save_path, folder_name)
    os.makedirs(save_folder)

    np.save(os.path.join(save_folder, 'result.npy'), result)
    
    with open(os.path.join(save_folder, 'flag.txt'), 'w') as f:
        f.write(*args)
    return 0

def save_in_time_hijack(result, ground_truth, save_path, *args):
    folder_name = 'output_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = os.path.join(save_path, folder_name)
    os.makedirs(save_folder)

    np.save(os.path.join(save_folder, 'result.npy'), result)
    np.save(os.path.join(save_folder, 'ground_truth.npy'), ground_truth)

    with open(os.path.join(save_folder, 'flag.txt'), 'w') as f:
        f.write(*args)
    return 0
