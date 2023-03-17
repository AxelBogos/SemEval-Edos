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

        train_all_tasks, dev_sets, test_sets = self.load_data(self.raw_data_dir)

        # Merge dev sets
        dev_sets_labelled = self.merge_labels_wrapper(dev_sets, full_data)

        # Merge test sets
        test_sets_labelled = self.merge_labels_wrapper(test_sets, full_data)

        # Encode train set targets
        train_all_tasks_encoded = self.encode_target(
            train_all_tasks, self._get_task_a_target_encoding, "label_sexist"
        )
        train_all_tasks_encoded = self.encode_target(
            train_all_tasks_encoded, self._get_task_b_target_encoding, "label_category"
        )
        train_all_tasks_encoded = self.encode_target(
            train_all_tasks_encoded, self._get_task_c_target_encoding, "label_vector"
        )

        # Encode dev sets targets
        dev_sets_labelled = self.encode_target_wrapper(dev_sets_labelled)

        # Encode test sets targets
        test_sets_labelled = self.encode_target_wrapper(test_sets_labelled)

        # Save encoded train set
        train_all_tasks_encoded.to_csv(
            Path(self.interim_output_dir, "train_all_tasks.csv"), index=False
        )

        # Save merged dev tests
        self.save_csv_wrapper(dev_sets_labelled, "dev", self.interim_output_dir)

        # Save merged test sets
        self.save_csv_wrapper(test_sets_labelled, "test", self.interim_output_dir)

    @staticmethod
    def merge(feature_df: pd.DataFrame, labels_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """The merge function takes two dataframes as input: a feature dataframe and a labels
        dataframe. It then merges the two on the 'rewire_id' column, which is common to both. The
        function then renames the label column from the labels df to 'target'. Finally, it returns
        this merged df.

        :param feature_df:pd.DataFrame: Specify the dataframe containing the features
        :param labels_df:pd.DataFrame: Specify the labels dataframe
        :param label_col:str: Specify which column in the labels_df contains the target variable
        :return: A dataframe with the features and labels merged on the rewire_id column
        """

        labels_df = labels_df[["rewire_id", label_col]]

        # Merge the dataframes on the common 'rewire_id' column
        merged_df = pd.merge(feature_df, labels_df, on="rewire_id", how="left")

        # rename label cols to standard 'target'
        merged_df = merged_df.rename(columns={label_col: "target"})

        # Return the merged dataframe
        return merged_df

    def merge_labels_wrapper(
        self,
        all_tasks_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        full_data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """The merge_labels_wrapper function takes in the dataframes for each task, and merges them
        with the full_data dataframe.

        :param self: Access the attributes of the class
        :param all_tasks_data:Tuple[pd.DataFrame: Pass the data from all tasks to the merge_labels_wrapper function
        :param full_data:pd.DataFrame: Pass the full dataset with all labels to the merge function
        :return: A tuple of 3 dataframes
        """
        task_a_set, task_b_set, task_c_set = all_tasks_data
        task_a_labelled = self.merge(task_a_set, full_data, "label_sexist")
        task_b_labelled = self.merge(task_b_set, full_data, "label_category")
        task_c_labelled = self.merge(task_c_set, full_data, "label_vector")
        return task_a_labelled, task_b_labelled, task_c_labelled

    @staticmethod
    def save_csv_wrapper(all_tasks_data, set_prefix: str, output_dir: str) -> None:
        """The save_csv_wrapper function saves the dataframes in all_tasks_data to csv files.

        :param all_tasks_data: Store the dataframes for each task
        :param set_prefix:str: a string that is used as part of the filename when saving the csv file. "dev" or "test".
        :param output_dir:str: Specify the path to where the data should be saved
        :return: None
        """
        task_a_set, task_b_set, task_c_set = all_tasks_data
        task_a_set.to_csv(Path(output_dir, f"{set_prefix}_task_a_entries.csv"), index=False)
        task_b_set.to_csv(Path(output_dir, f"{set_prefix}_task_b_entries.csv"), index=False)
        task_c_set.to_csv(Path(output_dir, f"{set_prefix}_task_c_entries.csv"), index=False)

    @staticmethod
    def encode_target(
        df: pd.DataFrame, encoding_dict: dict, target: str = "target"
    ) -> pd.DataFrame:
        """The encode_target function takes a DataFrame and a dictionary mapping the target values
        to integers. It then replaces the target column with an encoded version of itself, using
        the encoding_dict. The function returns this new DataFrame.

        :param df:pd.DataFrame: Specify the dataframe that we want to encode
        :param encoding_dict:dict: Dict mapping of the target column to integer
        :param target:str='target': Specify the name of the target column
        :return: The dataframe with the target column encoded
        """
        df[target].replace(encoding_dict, inplace=True)
        df[target] = df[target].astype(int)
        return df

    def encode_target_wrapper(
        self, all_tasks_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """The encode_target_wrapper function is a wrapper function that takes in the data for all
        three tasks, and returns the encoded dataframes for each task. This is done by calling the
        encode_target function on each of the tasks individually and then returning them as a
        tuple.

        :param self: Access the attributes and methods of the class in python
        :param all_tasks_data:Tuple[pd.DataFrame,..]: Pass the dataframes that contain the training, validation and test sets
        :return: The encoded dataframes for the 3 tasks
        """
        task_a_set, task_b_set, task_c_set = all_tasks_data
        task_a_set_encoded = self.encode_target(task_a_set, self._get_task_a_target_encoding)
        task_b_set_encoded = self.encode_target(task_b_set, self._get_task_b_target_encoding)
        task_c_set_encoded = self.encode_target(task_c_set, self._get_task_c_target_encoding)
        return (task_a_set_encoded, task_b_set_encoded, task_c_set_encoded)

    @staticmethod
    def load_data(
        data_root: str,
    ) -> Tuple[
        pd.DataFrame,
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ]:
        """The load_data function loads the data from the specified directory. It returns a tuple
        of three elements: 1) The training set, which is a Pandas DataFrame containing 2) The val
        sets of each task which is a tuple of 3 dataframes 3) the tests set of each task which is a
        tuple of 3 dataframes.

        :param data_root:str: Specify the path to the directory where our data is stored
        :return: Tuple[pd.DataFrame, Tuple, Tuple]
        """

        # Load train set
        train_set = pd.read_csv(Path(data_root, "train_all_tasks.csv"))

        # Load dev sets
        dev_task_a = pd.read_csv(Path(data_root, "dev_task_a_entries.csv"))
        dev_task_b = pd.read_csv(Path(data_root, "dev_task_b_entries.csv"))
        dev_task_c = pd.read_csv(Path(data_root, "dev_task_c_entries.csv"))

        # Load test sets
        test_task_a = pd.read_csv(Path(data_root, "test_task_a_entries.csv"))
        test_task_b = pd.read_csv(Path(data_root, "test_task_b_entries.csv"))
        test_task_c = pd.read_csv(Path(data_root, "test_task_c_entries.csv"))

        return (
            train_set,
            (dev_task_a, dev_task_b, dev_task_c),
            (test_task_a, test_task_b, test_task_c),
        )

    @property
    def _get_task_a_target_encoding(self) -> dict:
        """The _get_task_a_target_encoding function returns a dictionary mapping the target labels
        for task A to numbers.

        :param self: Access the attributes and methods of the class
        :return: A dictionary that maps the target labels to integers
        """
        return {"not sexist": 0, "sexist": 1}

    @property
    def _get_task_b_target_encoding(self) -> dict:
        """The _get_task_b_target_encoding function returns a dictionary mapping the target labels
        for task B to numbers.

        :param self: Access the attributes and methods of the class
        :return: A dictionary that maps the target labels to integers
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
        """The _get_task_b_target_encoding function returns a dictionary mapping the target labels
        for task C to numbers.

        :param self: Access the attributes and methods of the class
        :return: A dictionary that maps the target labels to integers
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
