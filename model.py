import torch
from transformers import BertModel, BertTokenizer, BertConfig
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
from torch.nn.functional import one_hot
from torch import nn
import numpy as np


class BERTPredictor(nn.Module):
    def __init__(self):
        super(BERTPredictor, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        # num classes
        self.linear1 = torch.nn.Linear(768, 3)
        self.relu = nn.ReLU()
        # self.weights = torch.FloatTensor(weights).to(device)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input_ids, attention_mask):
        _, encodings = self.model(input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(encodings)
        linear_output = self.linear1(dropout_output)
        sigmoid_layer = self.sigmoid(linear_output)
        final_layer = self.softmax(sigmoid_layer)
        return final_layer