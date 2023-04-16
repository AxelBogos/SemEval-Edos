from pathlib import Path
from typing import Optional

# import pytorch_lightning as pl
import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.components.Dataset import GenericDatasetTransformer
from src.data.text_processing import TextPreprocessor
from src.utils import defines


class DataModuleTransformerLocal(pl.LightningDataModule):
    def __init__(
        self,
        subtask,
        model="distilroberta-base",
        preprocessing_mode="none",
        batch_size=16,
        num_workers=1,
    ):
        """The __init__ function is called when the class is instantiated. It sets up the instance
        of the class, and defines all its attributes.

        :param self: Represent the instance of the class
        :param args: Pass the arguments from the command line to the class
        :return: None
        """
        super().__init__()
        self.model = model
        self.preprocessing_mode = preprocessing_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_train: Optional[GenericDatasetTransformer] = None
        self.data_val: Optional[GenericDatasetTransformer] = None
        self.data_test: Optional[GenericDatasetTransformer] = None

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.subtask = subtask

        # data preparation handlers
        self.text_preprocessor = TextPreprocessor(preprocessing_mode=self.preprocessing_mode)

    def setup(self, stage: Optional[str] = None):

        """The setup function is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split twice!

        :param self: Represent the instance of the class
        :param stage: Optional[str]: Determine whether the training is in train or test mode
        :return: A tuple of the train, val and test datasets
        """

        if not self.data_train:
            interim_data_train = pd.read_csv(Path(defines.INTERIM_DATA_DIR, "train.csv"))
            interim_data_train["text"] = self.text_preprocessor.transform_series(
                interim_data_train["text"]
            )
            interim_data_train = interim_data_train[
                interim_data_train[self._target_label].isin(self._get_subtask_targets)
            ]
            interim_data_train = interim_data_train.to_numpy()

            self.data_train = GenericDatasetTransformer(
                texts=interim_data_train[:, 1],
                labels=interim_data_train[:, self._target_index],
                tokenizer=self.tokenizer,
                max_token_len=128,
            )

        if not self.data_val:
            val_path = Path(defines.INTERIM_DATA_DIR, "val.csv")
            interim_data_val = pd.read_csv(val_path)
            interim_data_val["text"] = self.text_preprocessor.transform_series(
                interim_data_val["text"]
            )
            interim_data_val = interim_data_val[
                interim_data_val[self._target_label].isin(self._get_subtask_targets)
            ]
            interim_data_val = interim_data_val.to_numpy()

            self.data_val = GenericDatasetTransformer(
                texts=interim_data_val[:, 1],
                labels=interim_data_val[:, self._target_index],
                tokenizer=self.tokenizer,
                max_token_len=128,
            )

        if not self.data_test:
            interim_data_test = pd.read_csv(Path(defines.INTERIM_DATA_DIR, "test.csv"))
            interim_data_test["text"] = self.text_preprocessor.transform_series(
                interim_data_test["text"]
            )
            interim_data_test = interim_data_test[
                interim_data_test[self._target_label].isin(self._get_subtask_targets)
            ]
            interim_data_test = interim_data_test.to_numpy()

            self.data_test = GenericDatasetTransformer(
                texts=interim_data_test[:, 1],
                labels=interim_data_test[:, self._target_index],
                tokenizer=self.tokenizer,
                max_token_len=128,
            )

    def train_dataloader(self):
        """The train_dataloader function is used to load the training data.

        :param self: Represent the instance of the class
        :return: A dataloader object that contains the training data
        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """The val_dataloader function is used to create a DataLoader object for the validation
        set. The function takes no arguments and returns a DataLoader object that can be used to
        iterate over the validation data.

        :param self: Bind the instance of the class to the method
        :return: A dataloader object that contains the validation data
        """
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """The test_dataloader function is used to load the test data.

        :param self: Bind the attributes and methods of a class to the class object
        :return: A dataloader object
        """
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def _num_classes(self):

        """The _num_classes function returns the number of classes for a given task.

        :param self: Represent the instance of the class
        :return: The number of classes in the dataset
        """
        return len(self._get_subtask_targets)

    @property
    def _target_index(self):

        """The _target_index function returns the index of the target_col column in a training
        dataframe.

        :param self: Bind the instance of the class to a function
        :return: The index of the target_col column in the training data
        """
        if "a" in self.subtask:
            return 2
        elif "b" in self.subtask:
            return 3
        elif "c" in self.subtask:
            return 4

    @property
    def _target_label(self):

        """The _target_label function is used to determine the target_col label for training. The
        function takes in a single argument, self, which is an instance of the class.

        :param self: Bind the instance of the class to the method
        :return: The label of the training set for current task
        """
        if "a" in self.subtask:
            return "target_a"
        elif "b" in self.subtask:
            return "target_b"
        elif "c" in self.subtask:
            return "target_c"

    @property
    def _get_subtask_targets(self):
        if self.subtask == "a":
            return [0, 1]
        elif self.subtask == "b":
            return [0, 1, 2, 3]
        elif self.subtask == "c1":
            return [0, 1]
        elif self.subtask == "c2":
            return [2, 3, 4]
        elif self.subtask == "c3":
            return [5, 6, 7, 8]
        elif self.subtask == "c4":
            return [9, 10]
