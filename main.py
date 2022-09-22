import preprocessing
import numpy as np
from transformers import AutoTokenizer



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #preprocessing.preprocess_whole_set('both')
    train_data = np.load('preprocessed/english/score_train.npy', allow_pickle=True)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    for seq in train_data[1]:
        decoded_text = tokenizer.decode(seq)
        print(decoded_text)
    print(train_data[1][-1])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
