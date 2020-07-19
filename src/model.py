import config
import transformers
import torch.nn as nn
import torch
from torch.nn import functional as F

class NEWSclassifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, embedding_matrix,vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        
        #Constructor
        super().__init__()          
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        # self.embedding_dropout = SpatialDropout(0.3)

        # self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        # self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        # self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        # self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        # self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        # self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        
        
        
        # self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
    
        
        #dense layer
        extended_hidden_dim = hidden_dim * 4
        self.linear1 = nn.Linear(extended_hidden_dim, extended_hidden_dim)
        self.linear2 = nn.Linear(extended_hidden_dim, extended_hidden_dim)
        
        self.linear_out = nn.Linear(extended_hidden_dim, 1)

        # self.fc = nn.Linear(hidden_dim , output_dim)
        
        #activation function
        self.act = nn.Sigmoid()
        
       
    def forward(self, text):

        # print('*********',text.shape)
        
        #text = [batch size,sent_length]        
        embedded = self.embedding(text)
        # embedded = self.embedding_dropout(embedded)
        #embedded = [batch size, sent_len, emb dim]
        h_lstm1, _ = self.lstm(embedded)
        # h_lstm2, _ = self.lstm2(h_lstm1)

        # global average poolingy
        avg_pool = torch.mean(h_lstm1, 1)
        # # global max pooling
        max_pool, _ = torch.max(h_lstm1, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        dense_outputs = self.linear_out(hidden)


        # dense_outputs=self.fc(lstm_out).squeeze()

        #Final activation function
        outputs=self.act(dense_outputs) 
        # print('****************',outputs)       
        return outputs
    
        # def forward(self, sentence):
        # embeds = self.word_embeddings(sentence)
        # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores


        