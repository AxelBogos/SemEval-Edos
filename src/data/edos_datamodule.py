from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.Dataset import GenericDataset
from src.data.text_processing import SpacyTokenizer, TextPreprocessor


class EDOSDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.args = args

        # data preparation handlers
        self.text_preprocessor = TextPreprocessor(preprocessing_mode=self.args.preprocessing_mode)
        self.tokenizer = SpacyTokenizer()

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        """The setup function is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split twice!

        :param self: Represent the instance of the class
        :param stage: Optional[str]: Determine whether the training is in train or test mode
        :return: A tuple of the train, val and test datasets
        """

        if not self.data_train:
            train_path = Path(self.args.interim_data_dir, "train_all_tasks.csv")
            raw_data_train = pd.read_csv(train_path).to_numpy()
            self.data_train = GenericDataset(
                raw_data_train[:, [1, self._train_target_index]],
                self.tokenizer,
                self.text_preprocessor,
                self.args.max_length,
            )
        if not self.data_val:
            val_path = Path(self.args.interim_data_dir, f"dev_task_{self.args.task}.csv")
            raw_data_val = pd.read_csv(val_path).to_numpy()
            self.data_val = GenericDataset(
                raw_data_val[:, [1, 2]],
                self.tokenizer,
                self.text_preprocessor,
                self.args.max_length,
            )
        if not self.data_test:
            test_path = Path(self.args.interim_data_dir, f"test_task_{self.args.task}.csv")
            raw_data_test = pd.read_csv(test_path).to_numpy()
            self.data_test = GenericDataset(
                raw_data_test[:, [1, 2]],
                self.tokenizer,
                self.text_preprocessor,
                self.args.max_length,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
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
