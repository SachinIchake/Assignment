import config
import torch


class NEWSDataset:
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        label = self.label[item]        
        return {
            "text": torch.tensor(text, dtype=torch.long) ,
            "label": torch.tensor(label, dtype=torch.float) ,            
        }
