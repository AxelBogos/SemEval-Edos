from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

from src.data.augmentation_processing import AugmentationPreprocessor


class DataAugmentationProcessor:
    def __init__(self, interim_output_dir, augmented_output_dir):
        """The __init__ function is called when the class is instantiated. It sets up the instance
        variables for this particular object.

        :param self: Represent the instance of the class
        :param raw_data_dir: Specify the directory where the raw data is stored
        :param interim_output_dir: Specify the directory where the interim output files will be stored
        :return: Nothing
        """
        self.interim_output_dir = interim_output_dir
        self.augmented_output_dir = augmented_output_dir

    def run(self) -> None:
        """
        The run function is the entry point for this module. It does three things:
            1. Loads the data from a CSV file into a pandas DataFrame object
            2. Merges that DataFrame with another CSV file containing labels and other metadata
            3. Encode targets to integers (train,val test sets)
            3. Saves the merged datasets to disk as a new CSV file

        :param self: Access variables that belong to the class
        :return: Nothing, but it does write out the data to disk
        """

        # full_train_data = pd.read_csv(Path(self.interim_output_dir, "train.csv"))
        data_gab = pd.read_csv(Path(self.interim_output_dir, "task_b_GAB_aug.csv"))
        data_reddit = pd.read_csv(Path(self.interim_output_dir, "task_b_reddit_aug.csv"))

        full_train_data = pd.concat([data_gab, data_reddit])

        augmentation_bank = ["random_insertion_emb", "synonym_replacement_emb", "random_swap",
                             "random_deletion", "shuffle_sentence"]

        rewire_id_title = "rewire_id"
        target_label_a = "target_a"
        target_label_b = "target_b"
        target_label_c = "target_c"

        rewire_id, y_train_a, y_train_b, y_train_c = np.array(full_train_data[rewire_id_title]),\
                                                     np.array(full_train_data[target_label_a]), \
                                                     np.array(full_train_data[target_label_b]), \
                                                     np.array(full_train_data[target_label_c])

        for augmentation_type in augmentation_bank[:2]:
            data_augment_processor = self.get_data_augment_processor(augmentation_type)
            x_train = np.array(data_augment_processor.transform_series(full_train_data["text"]))

            train_df = pd.DataFrame({rewire_id_title: rewire_id, "text": x_train, target_label_a: y_train_a,
                                     target_label_b: y_train_b,
                                     target_label_c: y_train_c})

            train_df.to_csv(Path(self.augmented_output_dir, f"sf_train_augmented_{augmentation_type}_augmax_1.csv"), index=False)

    def get_data_augment_processor(self, augmentation_type):
        """The get_text_processor function returns a TextPreprocessor object with the following
        attributes:

            - lemmatize = False
            - tokenize = False

        :return: A textpreprocessor object
        """
        data_augment_processor = AugmentationPreprocessor(preprocessing_mode=augmentation_type)
        return data_augment_processor