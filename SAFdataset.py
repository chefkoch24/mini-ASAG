import torch
from torch.utils.data import Dataset


class SAF(Dataset):
    def __init__(self, data):
        self.dataset = {
            'question': data[0],
            'question_am': data[1],
            'student_answer': data[2],
            'student_answer_am': data[3],
            'reference_answer': data[4],
            'reference_answer_am': data[5],
            'response': data[6],
            'response_am': data[7],
            'label': data[8]
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.dataset.items()}
