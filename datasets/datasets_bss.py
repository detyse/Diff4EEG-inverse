import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class train_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.perturbed_path = r'/home/wyl/projects/_BSS_hijack/_end_to_end_compare/data/perturbed_data/perturbed.npy'
        self.truth_path = r'/home/wyl/projects/_BSS_hijack/_end_to_end_compare/data/separated_data/combined.npy'

        self.perturbed_data = np.load(self.perturbed_path)
        self.truth_data = np.load(self.truth_path)

    def __len__(self):
        len = self.perturbed_data.shape[0]
        return len
    
    def __getitem__(self, index):
        perturbed_data = self.perturbed_data[index]
        ground_truth = self.truth_data[index]
        return perturbed_data, ground_truth

class test_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.perturbed_path = r'/home/wyl/projects/_BSS_hijack/_end_to_end_compare/data/perturbed_data/perturbed_test.npy'
        self.truth_path = r'/home/wyl/projects/_BSS_hijack/_end_to_end_compare/data/separated_data/combined_test.npy'
        
        self.perturbed_data = np.load(self.perturbed_path)
        self.ground_truth = np.load(self.truth_path)
    
    def __len__(self):
        len = self.perturbed_data.shape[0]
        return len
    
    def __getitem__(self, index):
        perturbed_data = self.perturbed_data[index]
        ground_truth = self.ground_truth[index]
        return perturbed_data, ground_truth
    
'''
hijack dataset from trained data / for compare
'''