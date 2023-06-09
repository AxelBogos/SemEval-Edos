from pathlib import Path
from typing import Optional

import lightning as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.components.Dataset import GenericDatasetTransformer
from src.data.text_processing import TextPreprocessor


class DataModuleTransformerHierarichal(pl.LightningDataModule):
    def __init__(self, args):
        """The __init__ function is called when the class is instantiated. It sets up the instance
        of the class, and defines all its attributes.

        :param self: Represent the instance of the class
        :param args: Pass the arguments from the command line to the class
        :return: None
        """
        super().__init__()
        self.data_train: Optional[GenericDatasetTransformer] = None
        self.data_val: Optional[GenericDatasetTransformer] = None
        self.data_test: Optional[GenericDatasetTransformer] = None

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)

        # data preparation handlers
        self.text_preprocessor = TextPreprocessor(preprocessing_mode=self.args.preprocessing_mode)

    def setup(self, stage: Optional[str] = None):

        """The setup function is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split twice!

        :param self: Represent the instance of the class
        :param stage: Optional[str]: Determine whether the training is in train or test mode
        :return: A tuple of the train, val and test datasets
        """

        if not self.data_train:
            interim_data_train = pd.read_csv(Path(self.args.interim_data_dir, "train.csv"))
            interim_data_train["text"] = self.text_preprocessor.transform_series(
                interim_data_train["text"]
            )
            interim_data_train = interim_data_train.to_numpy()

            self.data_train = GenericDatasetTransformer(
                texts=interim_data_train[:, 1],
                labels=interim_data_train[:, [2, 3, 4]],
                tokenizer=self.tokenizer,
                max_token_len=self.args.max_token_length,
            )

        if not self.data_val:
            val_path = Path(self.args.interim_data_dir, "val.csv")
            interim_data_val = pd.read_csv(val_path)
            interim_data_val["text"] = self.text_preprocessor.transform_series(
                interim_data_val["text"]
            )

            # Add 1 to all labels
            interim_data_val = interim_data_val.to_numpy()

            self.data_val = GenericDatasetTransformer(
                texts=interim_data_val[:, 1],
                labels=interim_data_val[:, [2, 3, 4]],
                tokenizer=self.tokenizer,
                max_token_len=self.args.max_token_length,
            )

        if not self.data_test:
            interim_data_test = pd.read_csv(Path(self.args.interim_data_dir, "test.csv"))
            interim_data_test["text"] = self.text_preprocessor.transform_series(
                interim_data_test["text"]
            )
            interim_data_test = interim_data_test.to_numpy()

            self.data_test = GenericDatasetTransformer(
                texts=interim_data_test[:, 1],
                labels=interim_data_test[:, [2, 3, 4]],
                tokenizer=self.tokenizer,
                max_token_len=self.args.max_token_length,
            )

    def train_dataloader(self):
        """The train_dataloader function is used to load the training data.

        :param self: Represent the instance of the class
        :return: A dataloader object that contains the training data
        """
        return DataLoader(
            self.data_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        """The val_dataloader function is used to create a DataLoader object for the validation
        set. The function takes no arguments and returns a DataLoader object that can be used to
        iterate over the validation data.

        :param self: Bind the instance of the class to the method
        :return: A dataloader object that contains the validation data
        """
        return DataLoader(
            self.data_val,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        """The test_dataloader function is used to load the test data.

        :param self: Bind the attributes and methods of a class to the class object
        :return: A dataloader object
        """
        return DataLoader(
            self.data_test,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    @property
    def _num_classes(self):

        """The _num_classes function returns the number of classes for a given task.

        :param self: Represent the instance of the class
        :return: The number of classes in the dataset
        """
        return 0  # useless
