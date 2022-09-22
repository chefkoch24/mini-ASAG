import torch
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
from torch.nn.functional import one_hot
from torch import nn
import numpy as np


class BERTPredictor(nn.Module):
    def __init__(self):
        super(BERTPredictor, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        # num classes
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits