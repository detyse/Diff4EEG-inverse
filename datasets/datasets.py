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
        self.path = r'/home/wyl/projects/_BSS_hijack/data/separated_data/combined.npy'
        self.data = np.load(self.path)
        super().__init__()

    def __len__(self):
        len = self.data.shape[0]
        return len

    def __getitem__(self, index):
        get_trials = self.data[index]
        return get_trials
    
class separated_test_dataset(Dataset):
    def __init__(self) -> None:
        self.path = r'/home/wyl/projects/_BSS_hijack/data/separated_data/combined_test.npy'
        self.data = np.load(self.path)
        super().__init__()

    def __len__(self):
        len = self.data.shape[0]
        return len

    def __getitem__(self, index):
        get_trials = self.data[index]
        return get_trials