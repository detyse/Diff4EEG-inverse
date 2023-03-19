import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class perturbed_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = r'/home/wyl/projects/_BSS_hijack/data/perturbed_data/perturbed.npy'
        self.data = np.load(self.path)
    
    def __len__(self):
        len = self.data.shape[0]
        return len 

    def __getitem__(self, index):
        get_trials = self.data[index]
        return get_trials

class perturbed_test_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = r'/home/wyl/projects/_BSS_hijack/data/perturbed_data/perturbed_test.npy'
        self.data = np.load(self.path)

    def __len__(self):
        len = self.data.shape[0]
        return len
    
    def __getitem__(self, index):
        get_trials = self.data[index]
        return get_trials

class separated_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = r'/home/wyl/projects/_BSS_hijack/data/separated_data/combined.npy'
        self.data = np.load(self.path)

    def __len__(self):
        len = self.data.shape[0]
        return len

    def __getitem__(self, index):
        get_trials = self.data[index]
        return get_trials
    
class separated_test_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = r'/home/wyl/projects/_BSS_hijack/data/separated_data/combined_test.npy'
        self.data = np.load(self.path)

    def __len__(self):
        len = self.data.shape[0]
        return len

    def __getitem__(self, index):
        get_trials = self.data[index]
        return get_trials


class hijack_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.perturbed_path = r'/home/wyl/projects/_BSS_hijack/data/perturbed_data/perturbed_test.npy'
        self.truth_path = r'/home/wyl/projects/_BSS_hijack/data/separated_data/combined_test.npy'
        
        self.perturbed_data = np.load(self.perturbed_path)
        self.ground_truth = np.load(self.truth_path)

    def __len__(self):
        len = self.data.shape[0]
        return len
    
    def __getitem__(self, index):
        perturbed_data = self.perturbed_data[index]
        ground_truth = self.ground_truth[index]
        return perturbed_data, ground_truth
    
'''
hijack dataset from trained data / for compare
'''