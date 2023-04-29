from pathlib import Path
from typing import Optional

# import pytorch_lightning as pl
import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.components.Dataset import GenericDatasetTransformer
from src.data.text_processing import TextPreprocessor

from src.utils.defines import AUGMENTED_DATA_DIR
import os


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

        self.b1_aug_syn_ratio = args.b1_aug_syn_ratio
        self.b1_aug_insertion_ratio = args.b1_aug_insertion_ratio
        self.b1_sfr_syn_ratio = args.b1_sfr_syn_ratio
        self.b1_sfr_insertion_ratio = args.b1_sfr_insertion_ratio
        self.b1_aug_swap_ratio = args.b1_aug_swap_ratio

        self.b2_aug_insertion_ratio = args.b2_aug_insertion_ratio

        self.b3_aug_insertion_ratio = args.b3_aug_insertion_ratio

        self.b4_aug_syn_ratio = args.b4_aug_syn_ratio
        self.b4_aug_insertion_ratio = args.b4_aug_insertion_ratio
        self.b4_sfr_syn_ratio = args.b4_sfr_syn_ratio
        self.b4_sfr_insertion_ratio = args.b4_sfr_insertion_ratio
        self.b4_aug_swap_ratio = args.b4_aug_swap_ratio

        self.b1_sf_ratio = args.b1_sf
        self.b2_sf_ratio = args.b2_sf
        self.b3_sf_ratio = args.b3_sf
        self.b4_sf_ratio = args.b4_sf

        self.b1_sfr_ratio = args.b1_sfr
        self.b2_sfr_ratio = args.b2_sfr
        self.b3_sfr_ratio = args.b3_sfr
        self.b4_sfr_ratio = args.b4_sfr


    def setup(self, stage: Optional[str] = None):

        """The setup function is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split twice!

        :param self: Represent the instance of the class
        :param stage: Optional[str]: Determine whether the training is in train or test mode
        :return: A tuple of the train, val and test datasets
        """
        train_aug_insertion = pd.read_csv(os.path.join(AUGMENTED_DATA_DIR,
                                                       "train_augmented_random_insertion_emb_augmax_1.csv"))
        train_aug_synonym = pd.read_csv(os.path.join(AUGMENTED_DATA_DIR,
                                                     "train_augmented_synonym_replacement_emb.csv"))
        train_rand_swap = pd.read_csv(os.path.join(AUGMENTED_DATA_DIR, "train_augmented_random_swap.csv"))
        train_self_training = pd.read_csv(os.path.join(AUGMENTED_DATA_DIR, "task_b_GAB_aug.csv"))
        train_self_training_reddit = pd.read_csv(os.path.join(AUGMENTED_DATA_DIR, "task_b_reddit_aug.csv"))
        train_aug_insertion_sf = pd.read_csv(
            os.path.join(AUGMENTED_DATA_DIR, "sf_train_augmented_random_insertion_emb.csv"))
        train_aug_synonym_sf = pd.read_csv(
            os.path.join(AUGMENTED_DATA_DIR, "sf_train_augmented_synonym_replacement_emb.csv"))

        if not self.data_train:
            if self.args.task == 'a':
                interim_data_train = pd.read_csv(Path(self.args.interim_data_dir, "task_a_balanced.csv"))

            if self.args.task == 'b':
                interim_data_train = pd.read_csv(Path(self.args.interim_data_dir, "train.csv"))

                train_task_b = interim_data_train.copy()
                # train_task_b = train_task_b.loc[train_task_b['target_b'] != -1]

                b1 = train_task_b.loc[train_task_b['target_b'] == 0]
                b1_aug_syn = train_aug_synonym.loc[train_aug_synonym['target_b'] == 0]
                b1_aug_insertion = train_aug_insertion.loc[train_aug_insertion['target_b'] == 0]
                b1_aug_swap = train_rand_swap.loc[train_rand_swap['target_b'] == 0]
                b1_sf = train_self_training.loc[train_self_training['target_b'] == 0]
                b1_sfr = train_self_training_reddit.loc[train_self_training_reddit['target_b'] == 0]
                b1_sfr_insert = train_aug_insertion_sf.loc[train_aug_insertion_sf['target_b'] == 0]
                b1_sfr_syn = train_aug_synonym_sf.loc[train_aug_synonym_sf['target_b'] == 0]
                print(f"threats, plans to harm and incitement: {len(b1)}")

                b2 = train_task_b.loc[train_task_b['target_b'] == 1]
                b2_aug_syn = train_aug_synonym.loc[train_aug_synonym['target_b'] == 1]
                b2_aug_insertion = train_aug_insertion.loc[train_aug_insertion['target_b'] == 1]
                b2_sf = train_self_training.loc[train_self_training['target_b'] == 1]
                b2_sfr = train_self_training_reddit.loc[train_self_training_reddit['target_b'] == 1]
                b2_sfr_insert = train_aug_insertion_sf.loc[train_aug_insertion_sf['target_b'] == 1]
                b2_sfr_syn = train_aug_synonym_sf.loc[train_aug_synonym_sf['target_b'] == 1]
                print(f"derogation: {len(b2)}")

                b3 = train_task_b.loc[train_task_b['target_b'] == 2]
                b3_aug_syn = train_aug_synonym.loc[train_aug_synonym['target_b'] == 2]
                b3_aug_insertion = train_aug_insertion.loc[train_aug_insertion['target_b'] == 2]
                b3_sf = train_self_training.loc[train_self_training['target_b'] == 2]
                b3_sfr = train_self_training_reddit.loc[train_self_training_reddit['target_b'] == 2]
                b3_sfr_insert = train_aug_insertion_sf.loc[train_aug_insertion_sf['target_b'] == 2]
                b3_sfr_syn = train_aug_synonym_sf.loc[train_aug_synonym_sf['target_b'] == 2]
                print(f"animosity: {len(b3)}")

                b4 = train_task_b.loc[train_task_b['target_b'] == 3]
                b4_aug_syn = train_aug_synonym.loc[train_aug_synonym['target_b'] == 3]
                b4_aug_insertion = train_aug_insertion.loc[train_aug_insertion['target_b'] == 3]
                b4_aug_swap = train_rand_swap.loc[train_rand_swap['target_b'] == 3]
                b4_sf = train_self_training.loc[train_self_training['target_b'] == 3]
                b4_sfr = train_self_training_reddit.loc[train_self_training_reddit['target_b'] == 3]
                b4_sfr_insert = train_aug_insertion_sf.loc[train_aug_insertion_sf['target_b'] == 3]
                b4_sfr_syn = train_aug_synonym_sf.loc[train_aug_synonym_sf['target_b'] == 3]
                print(f"prejudiced discussions: {len(b4)}")

                print(f"total sexist task b: {len(pd.concat([b1, b2, b3, b4]))}")
                # dashing-surf-96
                # interim_data_train = pd.concat([train_task_b,
                #                                 b1_aug_insertion,
                #                                 b1_aug_syn.sample(int(len(b1) * 0.5)),
                #                                 b1_aug_swap.sample(int(len(b1) * 0)),
                #
                #                                 b2_aug_insertion.sample(int(len(b2) * 0)),
                #                                 b2_aug_syn.sample(int(len(b2) * 0)),
                #
                #                                 b3_aug_insertion.sample(int(len(b3) * 0)),
                #                                 b3_aug_syn.sample(int(len(b3) * 0)),
                #
                #                                 b4_aug_syn.sample(int(len(b4) * 0.5)),
                #                                 b4_aug_insertion.sample(int(len(b4) * 1)),
                #                                 b4_aug_swap.sample(int(len(b4) * 0)),
                #
                #                                 b1_sf.sample(int(len(b1_sf) * 1)),
                #                                 b2_sf.sample(int(len(b2_sf) * 0.4)),
                #                                 b3_sf.sample(int(len(b3_sf) * 0.2)),
                #                                 b4_sf.sample(int(len(b4_sf) * 1))
                #                                 ])

                interim_data_train = pd.concat([train_task_b,
                                                b1_aug_insertion.sample(int(len(b1) * self.b1_aug_insertion_ratio)),
                                                b1_aug_syn.sample(int(len(b1) * self.b1_aug_syn_ratio)),
                                                b1_aug_swap.sample(int(len(b1) * self.b1_aug_swap_ratio)),

                                                b2_aug_insertion.sample(int(len(b2) * self.b2_aug_insertion_ratio)),

                                                b3_aug_insertion.sample(int(len(b3) * self.b3_aug_insertion_ratio)),

                                                b4_aug_syn.sample(int(len(b4) * self.b4_aug_syn_ratio)),
                                                b4_aug_insertion.sample(int(len(b4) * self.b4_aug_insertion_ratio)),
                                                b4_aug_swap.sample(int(len(b4) * self.b4_aug_swap_ratio)),

                                                b1_sfr_insert.sample(int(len(b1_sfr_insert) * self.b1_sfr_insertion_ratio)),
                                                b1_sfr_syn.sample(int(len(b1_sfr_syn) * self.b1_sfr_syn_ratio)),

                                                b4_sfr_insert.sample(int(len(b4_sfr_insert) * self.b4_sfr_insertion_ratio)),
                                                b4_sfr_syn.sample(int(len(b4_sfr_syn) * self.b4_sfr_syn_ratio)),

                                                b1_sf.sample(int(len(b1_sf) * self.b1_sf_ratio)),
                                                b2_sf.sample(int(len(b2_sf) * self.b2_sf_ratio)),
                                                b3_sf.sample(int(len(b3_sf) * self.b3_sf_ratio)),
                                                b4_sf.sample(int(len(b4_sf) * self.b4_sf_ratio)),

                                                b1_sfr.sample(int(len(b1_sfr) * self.b1_sfr_ratio)),
                                                b2_sfr.sample(int(len(b2_sfr) * self.b2_sfr_ratio)),
                                                b3_sfr.sample(int(len(b3_sfr) * self.b3_sfr_ratio)),
                                                b4_sfr.sample(int(len(b4_sfr) * self.b4_sfr_ratio))
                                                ])
            if self.args.task == 'c':
                interim_data_train = pd.read_csv(Path(self.args.interim_data_dir, "train.csv"))

            interim_data_train["text"] = self.text_preprocessor.transform_series(
                interim_data_train["text"]
            )
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
