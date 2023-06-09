from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from src.data.components.Dataset import GenericDatasetLSTM
from src.data.text_processing import TextPreprocessor


class DataModuleLSTM(LightningDataModule):
    def __init__(self, args):

        """The __init__ function is called when the class is instantiated. It allows the class to
        initialize the attributes of a class. The self parameter is a reference to the current
        instance of the class, and is used to access variables that belong to the class.

        :param self: Represent the instance of the class
        :param args: Pass the arguments from the command line to the class
        :return: Nothing
        """
        super().__init__()

        self.data_train: Optional[GenericDatasetLSTM] = None
        self.data_val: Optional[GenericDatasetLSTM] = None
        self.data_test: Optional[GenericDatasetLSTM] = None

        self.args = args

        self.vocab = None
        self.collator = None
        self.pad_idx = None

        # data preparation handlers
        self.text_preprocessor = TextPreprocessor(preprocessing_mode=self.args.preprocessing_mode)

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
            interim_data_train = pd.read_csv(Path(self.args.interim_data_dir, "train.csv"))
            interim_data_train["text"] = self.text_preprocessor.transform_series(
                interim_data_train["text"]
            )
            interim_data_train = interim_data_train[interim_data_train[self._target_label] != 0]
            interim_data_train = interim_data_train.to_numpy()

            self.vocab = build_vocab_from_iterator(
                _helper_yield_tokens(interim_data_train), specials=["<unk>", "<pad>"], min_freq=5
            )
            self.vocab.set_default_index(self.vocab["<unk>"])
            self.collator = Collator(pad_idx=self.vocab["<pad>"])
            self.pad_idx = self.vocab["<pad>"]
            self.data_train = GenericDatasetLSTM(
                text=interim_data_train[:, 1],
                label=interim_data_train[:, self._target_index],
                vocab=self.vocab,
            )

        if not self.data_val:
            interim_data_val = pd.read_csv(Path(self.args.interim_data_dir, "val.csv"))
            interim_data_val["text"] = self.text_preprocessor.transform_series(
                interim_data_val["text"]
            )
            interim_data_val = interim_data_val[interim_data_val[self._target_label] != 0]
            interim_data_val = interim_data_val.to_numpy()

            self.data_val = GenericDatasetLSTM(
                text=interim_data_val[:, 1],
                label=interim_data_val[:, self._target_index],
                vocab=self.vocab,
            )

        if not self.data_test:
            interim_data_test = pd.read_csv(Path(self.args.interim_data_dir, "test.csv"))
            interim_data_test["text"] = self.text_preprocessor.transform_series(
                interim_data_test["text"]
            )
            interim_data_test = interim_data_test[interim_data_test[self._target_label] != 0]
            interim_data_test = interim_data_test.to_numpy()

            self.data_test = GenericDatasetLSTM(
                text=interim_data_test[:, 1],
                label=interim_data_test[:, self._target_index],
                vocab=self.vocab,
            )

    def train_dataloader(self):

        """The train_dataloader function is used to create a PyTorch DataLoader object for the
        training set.

        :param self: Bind the instance of the class to the method
        :return: A dataloader object which is used to load the data in batches
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            collate_fn=self.collator.collate,
        )

    def val_dataloader(self):
        """The val_dataloader function is used to create a DataLoader object for the validation
        set. The function takes no arguments and returns a DataLoader object that can be used to
        iterate over the validation set.

        :param self: Represent the instance of the class
        :return: A dataloader object
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=self.collator.collate,
        )

    def test_dataloader(self):

        """The test_dataloader function is used to create a DataLoader object for the test set.

        :param self: Bind the instance of the class to the method
        :return: A dataloader object that is used to iterate through the test dataset
        """
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
    def _target_index(self):

        """The _target_index function returns the index of the target_col column in a training
        dataframe.

        :param self: Bind the instance of the class to a function
        :return: The index of the target_col column in the training data
        """
        if self.args.task == "a":
            return 2
        elif self.args.task == "b":
            return 3
        elif self.args.task == "c":
            return 4

    @property
    def _target_label(self):

        """The _target_label function is used to determine the target_col label for training. The
        function takes in a single argument, self, which is an instance of the class.

        :param self: Bind the instance of the class to the method
        :return: The label of the training set for current task
        """
        if self.args.task == "a":
            return "target_a"
        elif self.args.task == "b":
            return "target_b"
        elif self.args.task == "c":
            return "target_c"


class Collator:
    def __init__(self, pad_idx):

        """The __init__ function is called when the class is instantiated. It sets up the instance
        of the class, and defines what attributes it has. In this case, we are setting up a
        Vocabulary object that will have an attribute pad_idx.

        :param self: Represent the instance of the class
        :param pad_idx: Determine which index in the vocabulary is used for padding
        :return: Nothing
        """
        self.pad_idx = pad_idx

    def collate(self, batch):

        """The collate function is used to batch together multiple samples.

        :param self: Access the attributes of the class
        :param batch: Pass the batch of data to the collate function
        :return: Padded sequences and labels
        """
        text, labels = zip(*batch)
        labels = torch.LongTensor(labels)
        text = torch.nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx, batch_first=True)
        return text, labels
