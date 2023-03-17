import numpy as np
import torch
from torch.utils.data import Dataset


class GenericDatasetLSTM(Dataset):
    def __init__(self, text: np.array, label: np.array, vocab=None):
        self.text = text
        self.label = label
        self.vocab = vocab

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        text, label = self.text[idx], self.label[idx]
        text = torch.tensor(self.vocab(text), dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return text, label
