import model
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import preprocessing
import numpy as np
from transformers import AutoTokenizer
import SAFdataset



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #preprocessing.preprocess_whole_set('both')
    train_data = np.load('preprocessed/english/score_train.npy', allow_pickle=True)
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #print(train_data[0])
    train_dataset = SAFdataset.SAFDataset(train_data)
    #print(train_dataset[1])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 1
    bert_pred = model.BERTPredictor()
    bert_pred.to(device)
    optimizer = torch.optim.AdamW(params=bert_pred.parameters(), lr=0.0001)
    bert_pred.train()
    for epoch in range(epochs):
        average_loss = []
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f'Epoch {epoch}')
        train_loss = 0
        for batch in loop:
            #input_ids = torch.cat((batch['student_answer'], batch['reference_answer'])).to(device)
            #attention_mask = torch.cat((batch['student_answer_attn'], batch['reference_answer_attn'])).to(device)
            input_ids = batch['student_answer'].to(device)
            attention_mask = batch['student_answer_attn'].to(device)
            # train model on batch and return outputs (incl. loss)
            targets = batch['label'].to(device)
            output = bert_pred.forward(input_ids, attention_mask)
            print(output)
            loss = torch.nn.BCELoss()(output, targets.float())
            optimizer.zero_grad()
            loss.backward()
            loop.set_postfix(loss=loss.item())
            average_loss.append(loss.item())
            optimizer.step()
        loop.set_postfix_str(s='average_loss=' + str(sum(average_loss) / len(average_loss)), refresh=True)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
