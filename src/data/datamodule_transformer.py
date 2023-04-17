from pathlib import Path
from typing import Optional

# import pytorch_lightning as pl
import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.components.Dataset import GenericDatasetTransformer
from src.data.text_processing import TextPreprocessor


class DataModuleTransformer(pl.LightningDataModule):
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

        # data augmentation experiments
        self.experiment = args.data_aug_exp
        self.set_experiment(args.data_aug_exp)

        # set data augmentations
        self.backtranslate_ratio = 0
        self.rand_deletion_ratio = 0
        self.rand_insertion_ratio = 0
        self.rand_swap_ratio = 0
        self.shuffle_sentence_ratio = 0
        self.syn_replacement_ratio = 0

    def setup(self, stage: Optional[str] = None):

        """The setup function is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split twice!

        :param self: Represent the instance of the class
        :param stage: Optional[str]: Determine whether the training is in train or test mode
        :return: A tuple of the train, val and test datasets
        """

        if not self.data_train:
            data_train = pd.read_csv(Path(self.args.interim_data_dir, "train.csv"))
            orig_len = len(data_train)

            augmented_data_train_backtranslation = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                    "train_augmented_backtranslate_all.csv"))
            augmented_data_train_rand_deletion = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                  "train_augmented_random_deletion.csv"))
            augmented_data_train_rand_insertion = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                   "train_augmented_random_insertion_emb.csv"))
            augmented_data_train_rand_swap = pd.read_csv(Path(self.args.augmented_data_dir,
                                                              "train_augmented_random_swap.csv"))
            augmented_data_train_shuffle_sentence = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                     "train_augmented_random_swap.csv"))
            augmented_data_train_syn_replacement = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                    "train_augmented_synonym_replacement_emb.csv"))
            interim_data_train = pd.concat([data_train,
                                            augmented_data_train_backtranslation.sample(orig_len*self.backtranslate_ratio),
                                            augmented_data_train_rand_deletion.sample(orig_len * self.rand_deletion_ratio),
                                            augmented_data_train_rand_insertion.sample(orig_len * self.rand_insertion_ratio),
                                            augmented_data_train_rand_swap.sample(orig_len * self.rand_swap_ratio),
                                            augmented_data_train_shuffle_sentence.sample(orig_len * self.shuffle_sentence_ratio),
                                            augmented_data_train_syn_replacement.sample(orig_len * self.syn_replacement_ratio),
                                            ])

            interim_data_train["text"] = self.text_preprocessor.transform_series(interim_data_train["text"])

            interim_data_train = interim_data_train[interim_data_train[self._target_label] != -1]
            interim_data_train = interim_data_train.to_numpy()

            self.data_train = GenericDatasetTransformer(
                texts=interim_data_train[:, 1],
                labels=interim_data_train[:, self._target_index],
                tokenizer=self.tokenizer,
                max_token_len=self.args.max_token_length,
            )

        if not self.data_val:
            val_path = Path(self.args.interim_data_dir, "val.csv")
            interim_data_val = pd.read_csv(val_path)
            interim_data_val["text"] = self.text_preprocessor.transform_series(
                interim_data_val["text"]
            )
            interim_data_val = interim_data_val[interim_data_val[self._target_label] != -1]
            interim_data_val = interim_data_val.to_numpy()

            self.data_val = GenericDatasetTransformer(
                texts=interim_data_val[:, 1],
                labels=interim_data_val[:, self._target_index],
                tokenizer=self.tokenizer,
                max_token_len=self.args.max_token_length,
            )

        if not self.data_test:
            interim_data_test = pd.read_csv(Path(self.args.interim_data_dir, "test.csv"))
            interim_data_test["text"] = self.text_preprocessor.transform_series(
                interim_data_test["text"]
            )
            interim_data_test = interim_data_test[interim_data_test[self._target_label] != -1]
            interim_data_test = interim_data_test.to_numpy()

            self.data_test = GenericDatasetTransformer(
                texts=interim_data_test[:, 1],
                labels=interim_data_test[:, self._target_index],
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

    def set_experiment(self, experiment: str) -> None:
        """

        """
        if experiment == "all":
            self.backtranslate_ratio = 0.1
            self.rand_deletion_ratio = 0.1
            self.rand_insertion_ratio = 0.1
            self.rand_swap_ratio = 0.1
            self.shuffle_sentence_ratio = 0.1
            self.syn_replacement_ratio = 0.1

        if experiment == "none":
            self.backtranslate_ratio = 0
            self.rand_deletion_ratio = 0
            self.rand_insertion_ratio = 0
            self.rand_swap_ratio = 0
            self.shuffle_sentence_ratio = 0
            self.syn_replacement_ratio = 0

        if experiment == "backtranslate":
            self.backtranslate_ratio = 0.5
            self.rand_deletion_ratio = 0
            self.rand_insertion_ratio = 0
            self.rand_swap_ratio = 0
            self.shuffle_sentence_ratio = 0
            self.syn_replacement_ratio = 0

        if experiment == "rand_deletion":
            self.backtranslate_ratio = 0
            self.rand_deletion_ratio = 0.5
            self.rand_insertion_ratio = 0
            self.rand_swap_ratio = 0
            self.shuffle_sentence_ratio = 0
            self.syn_replacement_ratio = 0

        if experiment == "rand_insertion":
            self.backtranslate_ratio = 0
            self.rand_deletion_ratio = 0
            self.rand_insertion_ratio = 0.5
            self.rand_swap_ratio = 0
            self.shuffle_sentence_ratio = 0
            self.syn_replacement_ratio = 0

        if experiment == "swap_ratio":
            self.backtranslate_ratio = 0
            self.rand_deletion_ratio = 0
            self.rand_insertion_ratio = 0
            self.rand_swap_ratio = 0.5
            self.shuffle_sentence_ratio = 0
            self.syn_replacement_ratio = 0

        if experiment == "shuffle_sentence":
            self.backtranslate_ratio = 0
            self.rand_deletion_ratio = 0
            self.rand_insertion_ratio = 0
            self.rand_swap_ratio = 0
            self.shuffle_sentence_ratio = 0.5
            self.syn_replacement_ratio = 0

        if experiment == "syn_replacement":
            self.backtranslate_ratio = 0
            self.rand_deletion_ratio = 0
            self.rand_insertion_ratio = 0
            self.rand_swap_ratio = 0
            self.shuffle_sentence_ratio = 0
            self.syn_replacement_ratio = 0.5
