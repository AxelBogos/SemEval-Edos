import re
import string
from pathlib import Path
from typing import Any

import pandas as pd
import pyrootutils
import spacy
import torch
from torchtext.vocab import GloVe


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
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS

        # Preprocessing flags.
        self.lower_case = False
        self.remove_multispace = False
        self.remove_punc = False
        self.remove_handles_and_urls = False
        self.remove_stop_words = False
        self.tokenize = False
        self.lemmatize = False

        self.set_preprocessing_flags(preprocessing_mode)

    def transform(self, x: str) -> str:
        """The transform function takes a string as input and returns a modified string. The
        transform function will:

            - lowercase the text, if self.lower_case is True;
            - remove Twitter handles and URLs, if self.remove_handles_and_urls is True;
            - replace multi whitespace to single spaces, if self.remove_multispace is True;
            - remove all punctuation from the text, if self.remove_punc is True;

        :param self: Allow the function to refer to and modify the attributes of the class
        :param x: String to be modified.
        :return: Modified string
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

        if preprocessing_mode == "bert":
            self.lower_case = False
            self.remove_multispace = False
            self.remove_punc = False
            self.remove_handles_and_urls = True
            self.remove_stop_words = True


class SpacyTokenizer:
    def __init__(self, embed_length=300, cache=Path):
        self.embed_length = embed_length
        self.nlp = spacy.load("en_core_web_sm")
        self.global_vectors = GloVe(
            name="840B",
            dim=embed_length,
            cache=Path(pyrootutils.find_root(), "data", ".vector_cache"),
        )

    def encode_plus(
        self,
        input_str: str,
        max_length: int,
        lemmatize: bool = True,
        truncation: bool = True,
        padding: str = "max_length",
        return_attention_mask: bool = True,
        return_tensors: Any = None,
    ) -> dict:
        """
        The encode_plus function is a tokenizing function that uses Spacy but follows the exepcted arguments of
         BertTokenizer.encode_plus:

        :param self: Access the variables and methods of the class in python
        :param input_str:str: Pass in the text that needs to be encoded
        :param max_length:int: Specify the maximum number of tokens that should be returned
        :param lemmatize:bool=True: Lemmatize the input string
        :param truncation:bool=True: Determine if the input string should be truncated or not
        :param padding:str="max_length": Specify the padding strategy. Useless, mimics Bert.tokenizer.
        :param return_attention_mask:bool=True: Return the attention mask (a binary tensor) along with the input_ids. Useless, mimics Bert.tokenizer.
        :param return_tensors:Any=None: Return the tensors in the format specified. Useless, mimics Bert.tokenizer.
        :return: A dictionary with the input_ids and attention_mask as keys
        """
        tokens = [token.lemma_ if lemmatize else token for token in self.nlp(input_str)]
        X_tensor = torch.zeros(max_length, self.embed_length)
        if truncation:
            tokens = tokens[:max_length]
        if len(tokens) < max_length:
            tokens = tokens + ["<pad>"] * (max_length - len(tokens))
        for idx, token in enumerate(tokens):
            X_tensor[idx] = self.global_vectors.get_vecs_by_tokens(token, lower_case_backup=True)
        attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        return {"input_ids": X_tensor, "attention_mask": attention_mask}
