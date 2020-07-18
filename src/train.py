import config
import dataset
import engine
from model import NEWSclassifier
import torch   
from torchtext import data 
import pandas as pd 

import numpy as np
from sklearn import model_selection
from sklearn import metrics
import torch.optim as optim
import torch.nn as nn 


def run():

    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none").reset_index(drop=True)
    # df_test = pd.read_csv(config.TESTING_FILE).fillna("none").reset_index(drop=True)

    df_train, df_test = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_dataset = dataset.NEWSDataset(
        text=df_train.text.values, label=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    test_dataset = dataset.NEWSDataset(
        text=df_test.text.values, label=df_test.label.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cpu")
    
    TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
    LABEL = data.LabelField(dtype = torch.float,batch_first=True)

    #initialize glove embeddings
    TEXT.build_vocab(df_train,min_freq=3,vectors = "glove.6B.100d")  
    LABEL.build_vocab(df_train)

    #No. of unique tokens in text
    print("Size of TEXT vocabulary:",len(TEXT.vocab))

    #No. of unique tokens in label
    print("Size of LABEL vocabulary:",len(LABEL.vocab))

    #Commonly used words
    print(TEXT.vocab.freqs.most_common(10))  

    #Word dictionary
    print(TEXT.vocab.stoi)   


    #define hyperparameters
    size_of_vocab = len(TEXT.vocab)
    embedding_dim = 100
    num_hidden_nodes = 32
    num_output_nodes = 1
    num_layers = 2
    bidirection = True
    dropout = 0.2

    #instantiate the model
    model = NEWSclassifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
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
