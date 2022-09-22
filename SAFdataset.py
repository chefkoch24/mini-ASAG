import torch
from torch.utils.data import Dataset


class SAFDataset(Dataset):
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        item = {'input_ids': data[0], 'attention_mask': data[1], 'token_type_ids': data[2], 'labels': data[3] }
        return item
