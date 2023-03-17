import re
import string
from pathlib import Path
from typing import Any

import pandas as pd
import spacy
import torch
from torchtext.vocab import GloVe

from src.utils import defines


class TextPreprocessor:
    """Preprocessing wrapper class. It implements the following:

    - def __init__(...):
        -Sets up the boolean internal states for which preprocessing to do
    - def transform(...):
        - Applies preprocessing to a string
    - def transform_series(...):
        - Applies preprocessing row-wise to a pd.Series
    """

    def __init__(self, preprocessing_mode: str = "standard") -> None:

        """The __init__ function is the constructor for a class. It is called when an object of
        that class is instantiated, and it sets up the attributes of that object. In this case, we
        are setting up the attributes for our Preprocessor object.

        :param self: Represent the instance of the class
        :param preprocessing_mode: str: Set the preprocessing flags
        :return: None
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS

        # Preprocessing flags.
        self.lower_case = False
        self.remove_multispace = False
        self.remove_punc = False
        self.remove_handles_and_urls = False
        self.remove_stop_words = False
        self.lemmatize = False

        self.set_preprocessing_flags(preprocessing_mode)

    def transform(self, x: str) -> list:
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
        if self.lower_case:
            x = x.lower()
        if self.remove_handles_and_urls:
            x = re.sub(r"(@[A-Za-z0-9]+)|(https?://[A-Za-z0-9./]+)|(\w+:\/\/\S+)", "", x)
        if self.remove_multispace:
            x = re.sub(r"\s+", " ", x)
        if self.remove_punc:
            x = re.sub("[" + string.punctuation + "]", "", x)
        if self.remove_stop_words:
            x = " ".join([word for word in x.split() if word not in self.stop_words])

        if self.lemmatize:
            x = [token.lemma_ for token in self.nlp(x)]
        else:
            x = [token for token in self.nlp(x)]

        return x

    def transform_series(self, series_col: pd.Series):
        """The transform_series function takes a pandas Series object and applies the transform
        function to each element in the series.

        :param self: Access the class attributes
        :param series_col:pd.Series: Specify the series that we want to transform
        :return: A series with the transform function applied to each element
        """
        series_col = series_col.apply(self.transform)
        return series_col

    def set_preprocessing_flags(self, preprocessing_mode: str) -> None:
        """The set_preprocessing_flags function sets the preprocessing flags for the class. The
        function takes in a string as an argument, which is used to set all of the preprocessing
        flags.

        :param self: Reference the class object
        :param preprocessing_mode:str: Set the preprocessing flags
        :return: None
        """
        if preprocessing_mode == "standard":
            self.lower_case = True
            self.remove_multispace = True
            self.remove_punc = True
            self.remove_handles_and_urls = True
            self.remove_stop_words = True
            self.lemmatize = True

        if preprocessing_mode == "none":
            self.lower_case = False
            self.remove_multispace = False
            self.remove_punc = False
            self.remove_handles_and_urls = False
            self.remove_stop_words = False
            self.lemmatize = False
