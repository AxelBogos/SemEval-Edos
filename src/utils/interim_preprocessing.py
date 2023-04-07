from pathlib import Path
from typing import Tuple

import pandas as pd


class InterimProcessor:
    def __init__(self, raw_data_dir, interim_output_dir):
        """The __init__ function is called when the class is instantiated. It sets up the instance
        variables for this particular object.

        :param self: Represent the instance of the class
        :param raw_data_dir: Specify the directory where the raw data is stored
        :param interim_output_dir: Specify the directory where the interim output files will be stored
        :return: Nothing
        """
        self.raw_data_dir = raw_data_dir
        self.interim_output_dir = interim_output_dir

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

        full_data = pd.read_csv(Path(self.raw_data_dir, "edos_labelled_aggregated.csv"))

        # Encode targets
        full_data = self.encode_target(full_data, self._get_task_a_target_encoding, "label_sexist")
        full_data = self.encode_target(
            full_data, self._get_task_b_target_encoding, "label_category"
        )
        full_data = self.encode_target(full_data, self._get_task_c_target_encoding, "label_vector")

        full_data = full_data.rename(
            columns={
                "label_sexist": "target_a",
                "label_category": "target_b",
                "label_vector": "target_c",
            }
        )

        # Split data into train, val and test sets
        train = full_data[full_data["split"] == "train"]
        val = full_data[full_data["split"] == "dev"]
        test = full_data[full_data["split"] == "test"]

        # Drop split column
        train = train.drop(columns=["split"])
        val = val.drop(columns=["split"])
        test = test.drop(columns=["split"])

        # Save train set
        train.to_csv(Path(self.interim_output_dir, "train.csv"), index=False)
        # Save val set
        val.to_csv(Path(self.interim_output_dir, "val.csv"), index=False)
        # Save test set
        test.to_csv(Path(self.interim_output_dir, "test.csv"), index=False)

    @staticmethod
    def encode_target(
        df: pd.DataFrame, encoding_dict: dict, target_col: str = "target_col"
    ) -> pd.DataFrame:
        """The encode_target function takes a DataFrame and a dictionary mapping the target values
        to integers. It then replaces the target column with an encoded version of itself, using
        the encoding_dict. The function returns this new DataFrame.

        :param df:pd.DataFrame: Specify the dataframe that we want to encode
        :param encoding_dict:dict: Dict mapping of the target column to integer
        :param target_col:str='target': Specify the name of the target column
        :return: The dataframe with the target column encoded
        """
        df[target_col].replace(encoding_dict, inplace=True)
        df[target_col] = df[target_col].astype(int)
        return df

    @property
    def _get_task_a_target_encoding(self) -> dict:
        """The _get_task_a_target_encoding function returns a dictionary mapping the target_col
        labels for task A to numbers.

        :param self: Access the attributes and methods of the class
        :return: A dictionary that maps the target_col labels to integers
        """
        return {"not sexist": 0, "sexist": 1}

    @property
    def _get_task_b_target_encoding(self) -> dict:
        """The _get_task_b_target_encoding function returns a dictionary mapping the target_col
        labels for task B to numbers.

        :param self: Access the attributes and methods of the class
        :return: A dictionary that maps the target_col labels to integers
        """
        return {
            "1. threats, plans to harm and incitement": 0,
            "2. derogation": 1,
            "3. animosity": 2,
            "4. prejudiced discussions": 3,
            "none": -1,
        }

    @property
    def _get_task_c_target_encoding(self) -> dict:
        """The _get_task_b_target_encoding function returns a dictionary mapping the target_col
        labels for task C to numbers.

        :param self: Access the attributes and methods of the class
        :return: A dictionary that maps the target_col labels to integers
        """
        return {
            "1.1 threats of harm": 0,
            "1.2 incitement and encouragement of harm": 1,
            "2.1 descriptive attacks": 2,
            "2.2 aggressive and emotive attacks": 3,
            "2.3 dehumanising attacks & overt sexual objectification": 4,
            "3.1 casual use of gendered slurs, profanities, and insults": 5,
            "3.2 immutable gender differences and gender stereotypes": 6,
            "3.3 backhanded gendered compliments": 7,
            "3.4 condescending explanations or unwelcome advice": 8,
            "4.1 supporting mistreatment of individual women": 9,
            "4.2 supporting systemic discrimination against women as a group": 10,
            "none": -1,
        }
