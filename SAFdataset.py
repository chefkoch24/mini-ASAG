import torch
from torch.utils.data import Dataset


class SAFDataset(Dataset):
    def __init__(self, data):

        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        question = data[0]
        question_am= data[1]
        student_answer = data[2]
        student_answer_am= data[3]
        reference_answer = data[4]
        reference_answer_am = data[5]
        response = data[6]
        response_am= data[7]
        label= data[8]
        return {'student_answer':torch.tensor(student_answer).long(), 'student_answer_attn':torch.tensor(student_answer_am).long(), 'reference_answer':torch.tensor(reference_answer).long(), 'reference_answer_attn':torch.tensor(reference_answer_am).long(), 'label': torch.tensor(label).long()}
        return torch.tensor(question).long(), torch.tensor(question_am).long(), torch.tensor(student_answer).long(), \
               torch.tensor(student_answer_am).long(), torch.tensor(reference_answer).long(), torch.tensor(reference_answer_am).long(), \
               torch.tensor(response).long(), torch.tensor(response_am).long(), torch.tensor(label).long()
