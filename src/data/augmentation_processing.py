import re
import string

import pandas as pd
import random
import spacy
import numpy as np
import nlpaug.augmenter.word as naw
from tqdm import tqdm
from multiprocessing import Pool


class AugmentationPreprocessor:
    """Preprocessing wrapper class. It implements the following:

    - def __init__(...):
        -Sets up the boolean internal states for which preprocessing to do
    - def transform(...):
        - Applies preprocessing to a string
    - def transform_series(...):
        - Applies preprocessing row-wise to a pd.Series
    """

    def __init__(self, preprocessing_mode: str = "synonym_replacement") -> None:

        """The __init__ function is the constructor for a class. It is called when an object of
        that class is instantiated, and it sets up the attributes of that object. In this case, we
        are setting up the attributes for our Preprocessor object.

        :param self: Represent the instance of the class
        :param preprocessing_mode: str: Set the preprocessing flags
        :return: None
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = self.nlp.Defaults.stop_words

        # Preprocessing flags.
        self.synonym_replacement = False
        self.random_insertion = False
        self.random_swap = False
        self.random_deletion = False
        self.shuffle_sentence = False

        self.preprocessing_mode = preprocessing_mode
        self.set_preprocessing_flags(preprocessing_mode)

        self.get_aug_substitute = naw.ContextualWordEmbsAug(model_path='roberta-large', action="substitute", aug_max=2)
        self.get_aug_insert = naw.ContextualWordEmbsAug(model_path='roberta-large', action="insert", aug_max=3)
        # self.get_aug_substitute = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute",
        #                                                     aug_max=1)
        # self.get_aug_insert = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="insert",
        #                                                 aug_max=1)

    def augment_data_experiment(self, x: str):
        """The transform function takes a string as input and returns a modified string. The
        transform function will:

            - lowercase the text, if self.lower_case is True;
            - remove Twitter handles and URLs, if self.remove_handles_and_urls is True;
            - replace multi whitespace to single spaces, if self.remove_multispace is True;
            - remove all punctuation from the text, if self.remove_punc is True;

        :param self: Allow the function to refer to and modify the attributes of the class
        :param x: String to be modified.
        :return: list of tokens
        """
        if self.synonym_replacement:
            x = self.get_aug_substitute.augment(x)[0]
        if self.random_insertion:
            x = self.get_aug_insert.augment(x)[0]
        if self.random_swap:
            x = self.get_random_swap(x, 1)
        if self.random_deletion:
            x = self.get_random_deletion(x, 0.1)
        if self.shuffle_sentence:
            x = self.shuffle_sentences(x)
        return x


    # def transform_series(self, series_col: pd.Series):
    #     """The transform_series function takes a pandas Series object and applies the transform
    #     function to each element in the series.
    #
    #     :param self: Access the class attributes
    #     :param series_col:pd.Series: Specify the series that we want to transform
    #     :return: A series with the transform function applied to each element
    #     """
    #     # series_col = series_col.apply(self.augment_data_experiment)
    #     # return series_col
    #     # Initialize the tqdm progress bar for pandas
    #     tqdm.pandas(desc=self.preprocessing_mode)
    #
    #     # Use progress_apply instead of apply to show the progress bar
    #     series_col = series_col.progress_apply(self.augment_data_experiment)
    #     return series_col

    @staticmethod
    def parallel_augment(args):
        instance, x = args
        return instance.augment_data_experiment(x)

    def transform_series(self, series_col: pd.Series, n_workers: int = 3):
        """The transform_series function takes a pandas Series object and applies the transform
        function to each element in the series.

        :param self: Access the class attributes
        :param series_col: pd.Series: Specify the series that we want to transform
        :param n_workers: int: Specify the number of worker processes to use for parallel processing
        :return: A series with the transform function applied to each element
        """

        with Pool(processes=n_workers) as pool:
            transformed_data = list(
                tqdm(pool.imap(AugmentationPreprocessor.parallel_augment, [(self, x) for x in series_col]),
                     total=len(series_col),
                     desc=self.preprocessing_mode))

        return pd.Series(transformed_data)

    def get_random_swap(self, sentence, n):
        words = sentence.split()
        length = len(words)

        if length < 2:
            sentence = ' '.join(words)
            return sentence

        for _ in range(n):
            i, j = random.randrange(length - 1), random.randrange(1, length)
            words[i], words[j] = words[j], words[i]

        sentence = ' '.join(words)
        return sentence

    def get_random_deletion(self, sentence, p=0.1):
        words = sentence.split()

        if len(words) == 1:
            return words

        remaining_words = [word for word in words if random.uniform(0, 1) > p]

        if len(remaining_words) == 0:
            return [random.choice(words)]
        else:
            sentence = ' '.join(remaining_words)
            return sentence

    def shuffle_sentences(self, sentence):
        # Regular expression pattern to match sentences
        sentence_pattern = r'[^.!?]+[.!?]'

        # Find all sentences in the input text
        sentences = re.findall(sentence_pattern, sentence)

        # Check if there are exactly three sentences
        if len(sentences) == 3:
            # Swap the first and second sentences, keeping the third sentence in its original position
            swapped_text = sentences[1].strip() + ' ' + sentences[0].strip() + ' ' + sentences[2].strip()
            return swapped_text
        if len(sentences) == 2:
            # Swap the sentences
            swapped_text = sentences[1].strip() + ' ' + sentences[0].strip()
            return swapped_text
        else:
            # Return the original text if the instance has more than 3 sentences
            return sentence

    def set_preprocessing_flags(self, preprocessing_mode: str) -> None:
        """The set_preprocessing_flags function sets the preprocessing flags for the class. The
        function takes in a string as an argument, which is used to set all of the preprocessing
        flags.

        :param self: Reference the class object
        :param preprocessing_mode:str: Set the preprocessing flags
        :return: None
        """

        if preprocessing_mode == "synonym_replacement_emb":
            self.synonym_replacement = True
            self.random_insertion = False
            self.random_swap = False
            self.random_deletion = False
            self.suffle_sentence = False

        if preprocessing_mode == "random_insertion_emb":
            self.synonym_replacement = False
            self.random_insertion = True
            self.random_swap = False
            self.random_deletion = False
            self.suffle_sentence = False

        if preprocessing_mode == "random_swap":
            self.synonym_replacement = False
            self.random_insertion = False
            self.random_swap = True
            self.random_deletion = False
            self.shuffle_sentence = False

        if preprocessing_mode == "random_deletion":
            self.synonym_replacement = False
            self.random_insertion = False
            self.random_swap = False
            self.random_deletion = True
            self.shuffle_sentence = False

        if preprocessing_mode == "shuffle_sentence":
            self.synonym_replacement = False
            self.random_insertion = False
            self.random_swap = False
            self.random_deletion = False
            self.shuffle_sentence = True

