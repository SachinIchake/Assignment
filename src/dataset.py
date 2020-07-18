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
        text = str(self.text[item])
        text = " ".join(text.split())
        label = str(self.label[item])
        # TODO TEXT PROCESSING
        
        return {
            "text": text,
            "label": label            
        }
