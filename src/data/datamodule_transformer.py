from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.components.Dataset import GenericDatasetTransformer
from src.data.text_processing import TextPreprocessor


class EDOSDataModuleTransformer(pl.LightningDataModule):
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
            train_path = Path(self.args.interim_data_dir, "train_all_tasks.csv")
            raw_data_train = pd.read_csv(train_path)
            raw_data_train["text"] = self.text_preprocessor.transform_series(
                raw_data_train["text"]
            )
            raw_data_train = raw_data_train[raw_data_train[self._train_target_label] != -1]
            raw_data_train = raw_data_train.to_numpy()

            self.data_train = GenericDatasetTransformer(
                texts=raw_data_train[:, 1],
                labels=raw_data_train[:, self._train_target_index],
                tokenizer=self.tokenizer,
                max_token_len=self.args.max_token_length,
            )

        if not self.data_val:
            val_path = Path(self.args.interim_data_dir, f"dev_task_{self.args.task}_entries.csv")
            raw_data_val = pd.read_csv(val_path)
            raw_data_val["text"] = self.text_preprocessor.transform_series(raw_data_val["text"])
            raw_data_val = raw_data_val.to_numpy()

            self.data_val = GenericDatasetTransformer(
                texts=raw_data_val[:, 1],
                labels=raw_data_val[:, 2],
                tokenizer=self.tokenizer,
                max_token_len=self.args.max_token_length,
            )

        if not self.data_test:
            test_path = Path(self.args.interim_data_dir, f"test_task_{self.args.task}_entries.csv")
            raw_data_test = pd.read_csv(test_path)
            raw_data_test["text"] = self.text_preprocessor.transform_series(raw_data_test["text"])
            raw_data_test = raw_data_test.to_numpy()

            self.data_test = GenericDatasetTransformer(
                texts=raw_data_test[:, 1],
                labels=raw_data_test[:, 2],
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
            self.data_val, batch_size=self.args.batch_size, num_workers=self.args.num_workers
        )

    def test_dataloader(self):
        """The test_dataloader function is used to load the test data.

        :param self: Bind the attributes and methods of a class to the class object
        :return: A dataloader object
        """
        return DataLoader(
            self.data_test, batch_size=self.args.batch_size, num_workers=self.args.num_workers
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

        """The _train_target_index function returns the index of the target_col column in a
        training dataframe.

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
    def _train_target_label(self):

        """The _train_target_label function is used to determine the target_col label for training.
        The function takes in a single argument, self, which is an instance of the class.

        :param self: Bind the instance of the class to the method
        :return: The label of the training set for current task
        """
        if self.args.task == "a":
            return "label_sexist"
        elif self.args.task == "b":
            return "label_category"
        elif self.args.task == "c":
            return "label_vector"
