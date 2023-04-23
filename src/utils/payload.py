from pathlib import Path
from typing import Optional
import os

# import pytorch_lightning as pl
import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.components.Dataset import GenericDatasetTransformer
from src.utils.defines import INTERIM_DATA_DIR, AUGMENTED_DATA_DIR
from src.data.text_processing import TextPreprocessor


class PayloadLoader:
    def __init__(self, args):
        self.args = args

        # set data augmentations
        self.b1_aug_backtranslate = 0
        self.b1_aug_swap = 0
        self.b1_aug_syn1_ratio = 0
        self.b1_aug_syn3_ratio = 0
        self.b1_aug_insert1_ratio = 0
        self.b1_aug_insert2_ratio = 0
        self.b1_aug_insert3_ratio = 0

        self.b2_aug_backtranslate = 0
        self.b2_aug_syn1_ratio = 0
        self.b2_aug_syn3_ratio = 0
        self.b2_aug_insert1_ratio = 0
        self.b2_aug_insert2_ratio = 0
        self.b2_aug_insert3_ratio = 0

        self.b3_aug_backtranslate = 0
        self.b3_aug_syn1_ratio = 0
        self.b3_aug_syn3_ratio = 0
        self.b3_aug_insert1_ratio = 0
        self.b3_aug_insert2_ratio = 0
        self.b3_aug_insert3_ratio = 0

        self.b4_aug_backtranslate = 0
        self.b4_aug_swap = 0
        self.b4_aug_syn1_ratio = 0
        self.b4_aug_syn3_ratio = 0
        self.b4_aug_insert1_ratio = 0
        self.b4_aug_insert2_ratio = 0
        self.b4_aug_insert3_ratio = 0

        # data augmentation experiments
        self.experiment = args.data_aug_exp
        self.set_experiment(args.data_aug_exp)

    def balanced_class(self):
        # load augmented set
        train_aug_backtranslate = pd.read_csv(Path(self.args.augmented_data_dir,
                                                "train_augmented_backtranslate_all.csv"))

        train_aug_insertion3 = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                "train_augmented_random_insertion_emb.csv"))
        train_aug_insertion1 = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                "train_augmented_random_insertion_emb_aug_max_1.csv"))
        train_aug_insertion2 = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                "train_augmented_random_insertion_emb_aug_max_2.csv"))

        train_rand_swap = pd.read_csv(Path(self.args.augmented_data_dir,
                                                          "train_augmented_random_swap.csv"))

        train_aug_synonym3 = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                 "train_augmented_synonym_replacement_emb.csv"))
        train_aug_synonym1 = pd.read_csv(Path(self.args.augmented_data_dir,
                                                                 "train_augmented_synonym_replacement_emb_augmax_1.csv"))


        if self.args.task == 'b':
            # Each sexist category size
            train_df = pd.read_csv(os.path.join(INTERIM_DATA_DIR, "train.csv"))

            train_task_b = train_df.copy()
            train_task_b = train_task_b.loc[train_task_b['target_b'] != -1]
            b1 = train_task_b.loc[train_task_b['target_b'] == 0]
            b2 = train_task_b.loc[train_task_b['target_b'] == 1]
            b3 = train_task_b.loc[train_task_b['target_b'] == 2]
            b4 = train_task_b.loc[train_task_b['target_b'] == 3]

            #####################################################################################
            # b1: threats, plans to harm and incitement
            b1_aug_backtranslate = train_aug_backtranslate.loc[train_aug_backtranslate['target_b'] == 0]

            b1_aug_syn1 = train_aug_synonym1.loc[train_aug_synonym1['target_b'] == 0]
            b1_aug_syn3 = train_aug_synonym3.loc[train_aug_synonym3['target_b'] == 0]

            b1_aug_insertion1 = train_aug_insertion1.loc[train_aug_insertion1['target_b'] == 0]
            b1_aug_insertion2 = train_aug_insertion2.loc[train_aug_insertion2['target_b'] == 0]
            b1_aug_insertion3 = train_aug_insertion3.loc[train_aug_insertion3['target_b'] == 0]

            b1_aug_swap = train_rand_swap.loc[train_rand_swap['target_b'] == 0]
            #####################################################################################
            # b2: derogation
            b2_aug_backtranslate = train_aug_backtranslate.loc[train_aug_backtranslate['target_b'] == 1]

            b2_aug_syn1 = train_aug_synonym1.loc[train_aug_synonym1['target_b'] == 1]
            b2_aug_syn3 = train_aug_synonym3.loc[train_aug_synonym3['target_b'] == 1]

            b2_aug_insertion1 = train_aug_insertion1.loc[train_aug_insertion1['target_b'] == 1]
            b2_aug_insertion2 = train_aug_insertion2.loc[train_aug_insertion2['target_b'] == 1]
            b2_aug_insertion3 = train_aug_insertion3.loc[train_aug_insertion3['target_b'] == 1]
            #####################################################################################
            # b3: animosity
            b3_aug_backtranslate = train_aug_backtranslate.loc[train_aug_backtranslate['target_b'] == 2]

            b3_aug_syn1 = train_aug_synonym1.loc[train_aug_synonym1['target_b'] == 2]
            b3_aug_syn3 = train_aug_synonym3.loc[train_aug_synonym3['target_b'] == 2]

            b3_aug_insertion1 = train_aug_insertion1.loc[train_aug_insertion1['target_b'] == 2]
            b3_aug_insertion2 = train_aug_insertion2.loc[train_aug_insertion2['target_b'] == 2]
            b3_aug_insertion3 = train_aug_insertion3.loc[train_aug_insertion3['target_b'] == 2]
            #####################################################################################
            # b4: prejudiced discussions
            b4_aug_backtranslate = train_aug_backtranslate.loc[train_aug_backtranslate['target_b'] == 3]

            b4_aug_syn1 = train_aug_synonym1.loc[train_aug_synonym1['target_b'] == 3]
            b4_aug_syn3 = train_aug_synonym3.loc[train_aug_synonym3['target_b'] == 3]

            b4_aug_insertion1 = train_aug_insertion1.loc[train_aug_insertion1['target_b'] == 3]
            b4_aug_insertion2 = train_aug_insertion2.loc[train_aug_insertion2['target_b'] == 3]
            b4_aug_insertion3 = train_aug_insertion3.loc[train_aug_insertion3['target_b'] == 3]
            
            b4_aug_swap = train_rand_swap.loc[train_rand_swap['target_b'] == 3]

            aug_data_train = pd.concat([
                b1_aug_backtranslate.sample(int(len(b1) * self.b1_aug_backtranslate)),
                b1_aug_swap.sample(int(len(b1) * self.b1_aug_swap)),
                b1_aug_syn1.sample(int(len(b1) * self.b1_aug_syn1_ratio)),
                b1_aug_syn3.sample(int(len(b1) * self.b1_aug_syn3_ratio)),
                b1_aug_insertion1.sample(int(len(b1) * self.b1_aug_insert1_ratio)),
                b1_aug_insertion2.sample(int(len(b1) * self.b1_aug_insert2_ratio)),
                b1_aug_insertion3.sample(int(len(b1) * self.b1_aug_insert3_ratio)),
                b2_aug_backtranslate.sample(int(len(b2) * self.b2_aug_backtranslate)),
                b2_aug_syn1.sample(int(len(b2) * self.b2_aug_syn1_ratio)),
                b2_aug_syn3.sample(int(len(b2) * self.b2_aug_syn3_ratio)),
                b2_aug_insertion1.sample(int(len(b2) * self.b2_aug_insert1_ratio)),
                b2_aug_insertion2.sample(int(len(b2) * self.b2_aug_insert2_ratio)),
                b2_aug_insertion3.sample(int(len(b2) * self.b2_aug_insert3_ratio)),
                b3_aug_backtranslate.sample(int(len(b3) * self.b3_aug_backtranslate)),
                b3_aug_syn1.sample(int(len(b3) * self.b3_aug_syn1_ratio)),
                b3_aug_syn3.sample(int(len(b3) * self.b3_aug_syn3_ratio)),
                b3_aug_insertion1.sample(int(len(b3) * self.b3_aug_insert1_ratio)),
                b3_aug_insertion2.sample(int(len(b3) * self.b3_aug_insert2_ratio)),
                b3_aug_insertion3.sample(int(len(b3) * self.b3_aug_insert3_ratio)),
                b4_aug_backtranslate.sample(int(len(b4) * self.b4_aug_backtranslate)),
                b4_aug_swap.sample(int(len(b4) * self.b4_aug_swap)),
                b4_aug_syn1.sample(int(len(b4) * self.b4_aug_syn1_ratio)),
                b4_aug_syn3.sample(int(len(b4) * self.b4_aug_syn3_ratio)),
                b4_aug_insertion1.sample(int(len(b4) * self.b4_aug_insert1_ratio)),
                b4_aug_insertion2.sample(int(len(b4) * self.b4_aug_insert2_ratio)),
                b4_aug_insertion3.sample(int(len(b4) * self.b4_aug_insert3_ratio)),
            ])
            return aug_data_train

    def set_experiment(self, experiment: str) -> None:
        """

        """
        if experiment == "task-b-none":
            # keep default ratio
            pass

        if experiment == "task-b-syn1-ins1":
            self.b1_aug_swap = 1
            self.b1_aug_syn1_ratio = 1
            self.b1_aug_insert1_ratio = 1

            self.b2_aug_insert1_ratio = 0.5

            self.b3_aug_insert1_ratio = 0.5

            self.b4_aug_swap = 1
            self.b4_aug_syn1_ratio = 1
            self.b4_aug_insert1_ratio = 1

        if experiment == "task-b-syn1-ins1-onlyless":
            self.b1_aug_swap = 1
            self.b1_aug_syn1_ratio = 1
            self.b1_aug_insert1_ratio = 1

            self.b4_aug_swap = 1
            self.b4_aug_syn1_ratio = 1
            self.b4_aug_insert1_ratio = 1

        if experiment == "task-b-syn1-ins1-backtrsl-onlyless":
            self.b1_aug_swap = 1
            self.b1_aug_syn1_ratio = 1
            self.b1_aug_insert1_ratio = 1
            self.b1_aug_backtranslate = 1

            self.b2_aug_backtranslate = 1

            self.b3_aug_backtranslate = 1

            self.b4_aug_swap = 1
            self.b4_aug_syn1_ratio = 1
            self.b4_aug_insert1_ratio = 1
            self.b4_aug_backtranslate = 1

        if experiment == "task-b-syn3-ins3":
            self.b1_aug_swap = 1
            self.b1_aug_syn3_ratio = 1
            self.b1_aug_insert3_ratio = 1

            self.b2_aug_insert3_ratio = 1

            self.b3_aug_insert3_ratio = 1

            self.b4_aug_swap = 1
            self.b4_aug_syn3_ratio = 1
            self.b4_aug_insert3_ratio = 1

        if experiment == "task-b-syn3-ins3-less":
            self.b1_aug_swap = 0.5
            self.b1_aug_syn3_ratio = 1
            self.b1_aug_insert3_ratio = 1

            self.b3_aug_insert3_ratio = 0.2

            self.b4_aug_swap = 0.5
            self.b4_aug_syn3_ratio = 1
            self.b4_aug_insert3_ratio = 1

        if experiment == "task-b-syn3-ins3-less-v2":
            self.b1_aug_swap = 1
            self.b1_aug_syn3_ratio = 1
            self.b1_aug_insert3_ratio = 1

            self.b2_aug_insert3_ratio = 0.2

            self.b3_aug_insert3_ratio = 0.5

            self.b4_aug_swap = 1
            self.b4_aug_syn3_ratio = 1
            self.b4_aug_insert3_ratio = 1

        if experiment == "task-b-syn3-ins3-onlyless":
            self.b1_aug_swap = 1
            self.b1_aug_syn3_ratio = 1
            self.b1_aug_insert3_ratio = 1

            self.b4_aug_swap = 1
            self.b4_aug_syn3_ratio = 1
            self.b4_aug_insert3_ratio = 1

        if experiment == "task-b-syn3-ins3-backtrsl-onlyless":
            self.b1_aug_swap = 1
            self.b1_aug_syn3_ratio = 1
            self.b1_aug_insert3_ratio = 1
            self.b1_aug_backtranslate = 1

            self.b2_aug_backtranslate = 1

            self.b3_aug_backtranslate = 1

            self.b4_aug_swap = 1
            self.b4_aug_syn3_ratio = 1
            self.b4_aug_insert3_ratio = 1
            self.b4_aug_backtranslate = 1

        if experiment == "task-b-syn1-ins1-backtrsl-mixed":
            self.b1_aug_swap = 1
            self.b1_aug_syn1_ratio = 1
            self.b1_aug_insert1_ratio = 1
            self.b1_aug_backtranslate = 1

            self.b2_aug_insert1_ratio = 0.2

            self.b3_aug_syn1_ratio = 0.25
            self.b3_aug_insert1_ratio = 0.25

            self.b4_aug_swap = 1
            self.b4_aug_syn1_ratio = 1
            self.b4_aug_insert1_ratio = 1
            self.b4_aug_backtranslate = 1

