from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

from src.data.components.Dataset import GenericDataset
from src.data.text_processing import TextPreprocessor


class EDOSDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.args = args

        self.vocab = None
        self.collator = None
        self.pad_idx = None

        # data preparation handlers
        self.text_preprocessor = TextPreprocessor(preprocessing_mode=self.args.preprocessing_mode)
        # self.tokenizer = SpacyTokenizer()

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        """The setup function is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split twice!

        :param self: Represent the instance of the class
        :param stage: Optional[str]: Determine whether the training is in train or test mode
        :return: A tuple of the train, val and test datasets
        """

        def _helper_yield_tokens(train_dataset):
            """The _helper_yield_tokens function is a generator that yields the tokens from each
            row in the train_dataset.

            :param train_dataset: Pass in the dataset that we want to tokenize
            :return: A generator object
            """
            for row in train_dataset:
                yield row[1]

        if not self.data_train:
            train_path = Path(self.args.interim_data_dir, "train_all_tasks.csv")
            raw_data_train = pd.read_csv(train_path)
            raw_data_train["text"] = self.text_preprocessor.transform_series(
                raw_data_train["text"]
            )
            raw_data_train = raw_data_train[raw_data_train[self._train_target_index] != -1]
            raw_data_train = raw_data_train.to_numpy()

            self.vocab = build_vocab_from_iterator(
                _helper_yield_tokens(raw_data_train), specials=["<unk>", "<pad>"], min_freq=5
            )
            self.vocab.set_default_index(self.vocab["<unk>"])
            self.collator = Collator(pad_idx=self.vocab["<pad>"])
            self.pad_idx = self.vocab["<pad>"]
            self.data_train = GenericDataset(
                text=raw_data_train[:, 1],
                label=raw_data_train[:, self._train_target_index],
                vocab=self.vocab,
            )

        if not self.data_val:
            val_path = Path(self.args.interim_data_dir, f"dev_task_{self.args.task}_entries.csv")
            raw_data_val = pd.read_csv(val_path)
            raw_data_val["text"] = self.text_preprocessor.transform_series(raw_data_val["text"])

            raw_data_val = raw_data_val.to_numpy()

            self.data_val = GenericDataset(
                text=raw_data_val[:, 1], label=raw_data_val[:, 2], vocab=self.vocab
            )

        if not self.data_test:
            test_path = Path(self.args.interim_data_dir, f"test_task_{self.args.task}_entries.csv")
            raw_data_test = pd.read_csv(test_path)
            raw_data_test["text"] = self.text_preprocessor.transform_series(raw_data_test["text"])
            raw_data_test = raw_data_test.to_numpy()
            self.data_test = GenericDataset(
                text=raw_data_test[:, 1], label=raw_data_test[:, 2], vocab=self.vocab
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            collate_fn=self.collator.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=self.collator.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=self.collator.collate,
        )

    @property
    def _num_classes(self):

        """The _num_classes function returns the number of classes for a given task.

        :param self: Represent the instance of the class
        :return: The number of classes in the dataset
        """
        if self.args.task == "a":
            return 2
        elif self.args.task == "b":
            return 4
        elif self.args.task == "c":
            return 11

    @property
    def _train_target_index(self):

        """The _train_target_index function returns the index of the target column in a training
        dataframe.

        :param self: Bind the instance of the class to a function
        :return: The index of the target column in the training data
        """
        if self.args.task == "a":
            return 2
        elif self.args.task == "b":
            return 3
        elif self.args.task == "c":
            return 4


class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def collate(self, batch):
        text, labels = zip(*batch)
        labels = torch.LongTensor(labels)
        text = torch.nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx, batch_first=True)
        return text, labels
