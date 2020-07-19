import config
import dataset
import engine
from model import NEWSclassifier
import torch   
from torchtext import data 
import pandas as pd 
from keras.preprocessing import text, sequence
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import torch.optim as optim
import torch.nn as nn 
from tqdm import tqdm
import os
import time
import gc
import random
from keras.preprocessing import text, sequence
import torch
from torch.utils import data
from torch.nn import functional as F
import config 


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path,encoding='utf-8') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words
    
def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data

def run():

    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none").reset_index(drop=True)
    # df_test = pd.read_csv(config.TESTING_FILE).fillna("none").reset_index(drop=True)

    df_train, df_test = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    x_train = preprocess(df_train['text'])
    y_train = df_train['label']
    x_test = preprocess(df_test['text'])
    y_test = df_test['label']

    max_features = None
    MAX_LEN = 400

    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(x_train))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    max_features = max_features or len(tokenizer.word_index) + 1

    glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, config.GLOVE)
    print('n unknown words (glove): ', len(unknown_words_glove))

    embedding_matrix = glove_matrix


    del glove_matrix
    gc.collect()

    # x_train_torch = torch.tensor(x_train, dtype=torch.long)
    # y_train_torch = torch.tensor(y_train, dtype=torch.int32)
    # x_test_torch = torch.tensor(x_train, dtype=torch.long)
    # y_test_torch = torch.tensor(y_train, dtype=torch.int32)


    train_dataset = dataset.NEWSDataset(
        text=x_train, label=y_train
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    test_dataset = dataset.NEWSDataset(
        text=x_test, label=y_test
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cpu")
    
   
    #define hyperparameters
    size_of_vocab = embedding_matrix.shape[1]
    embedding_dim = 300
    num_hidden_nodes = 128
    num_output_nodes = 1
    num_layers = 2
    bidirection = True
    dropout = 0.2

    #instantiate the model
    model = NEWSclassifier(embedding_matrix,size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                    bidirectional = True, dropout = dropout)

    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = optim.Adam(model.parameters())
    

    model = nn.DataParallel(model)


    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device)
        outputs, labels = engine.eval_fn(test_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(labels, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == "__main__":
    run()
