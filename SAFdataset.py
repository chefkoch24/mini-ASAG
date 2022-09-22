import torch
from torch.utils.data import Dataset


class SAFDataset(Dataset):
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        item = {'input_ids': torch.tensor(data[0]).long(), 'attention_mask': torch.tensor(data[1]).long(), 'token_type_ids': torch.tensor(data[2]).long(), 'labels': torch.tensor(data[3]).long() }
        return item
