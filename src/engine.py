import torch
import torch.nn as nn
from tqdm import tqdm
from model import NEWSclassifier

def loss_fn(outputs, labels):
    return nn.BCEWithLogitsLoss()(outputs, labels.view(-1, 1))


def train_fn(data_loader, model, optimizer, device):
                    
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        text = d["text"]
        labels = d["label"]
        
        
        # text = text.to(device, dtype=torch.long)
        # label = label.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(text)
        # print(outputs.shape)
        # print(labels.shape)


        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        

def eval_fn(data_loader, model, device):
    model.eval()
    fin_labels = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            text = d["text"]
            labels = d["label"]
            
            # text = text.to(device, dtype=torch.long)
            # label = label.to(device, dtype=torch.long)
             
            outputs = model(text)
            fin_labels.extend(labels.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_labels
