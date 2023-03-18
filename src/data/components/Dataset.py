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


class GenericDatasetTransformer(Dataset):
    def __init__(
        self,
        texts: np.array,
        labels: np.array,
        tokenizer,
        max_token_len: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(label, dtype=torch.long),
        )
