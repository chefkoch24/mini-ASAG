import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.model_selection import train_test_split
import preprocessing
import SAFdataset
import model
from tqdm import tqdm



def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #preprocessing.preprocess_whole_set('both')
    train_data = np.load('preprocessed/english/score_train.npy', allow_pickle=True)
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = SAFdataset.SAFDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 1
    bert_pred = model.BERTPredictor()
    bert_pred.to(device)
    optimizer = torch.optim.AdamW(params=bert_pred.parameters(), lr=0.0001)
    bert_pred.train()
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        average_loss = []
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f'Epoch {epoch}')
        train_loss = 0
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            # train model on batch and return outputs (incl. loss)
            targets = batch['labels'].to(device)
            # print(targets)
            output = bert_pred.forward(input_ids, attention_mask, token_type_ids)
            # print(output)
            loss = criterion(output.view(-1, 3), targets.view(-1))
            loss.backward()
            loop.set_postfix(loss=loss.item())
            average_loss.append(loss.item())
            optimizer.step()
        print('average_loss=', str(sum(average_loss) / len(average_loss)))


