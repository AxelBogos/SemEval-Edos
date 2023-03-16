from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.text_processing import SpacyTokenizer, TextPreprocessor


class GenericDataset(Dataset):
    def __init__(
        self,
        data: np.array,
        tokenizer: Union[SpacyTokenizer],  # Include Bert.Tokenizer too eventually
        preprocessor: TextPreprocessor,
        max_length: int,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        text = self.preprocessor.transform(text)
        encoded_text = self.tokenizer.encode_plus(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"]
        attention_mask = encoded_text["attention_mask"]
        return input_ids, torch.Tensor(attention_mask), torch.Tensor([label])
