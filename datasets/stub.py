import torch
from torch.utils.data import Dataset


class Stub(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'image': torch.randn((3, 128, 128)) * 2 - 1, 'target': torch.rand((10, 19))}
