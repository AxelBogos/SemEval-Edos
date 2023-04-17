import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from multiprocessing import Pool

from src.utils import defines


def main() -> None:
    """The main function of this module. Runs a multinomial-GNB baseline on all tasks.
    :return: Nothing
    """

    # Load train
    train = pd.read_csv(Path(defines.INTERIM_DATA_DIR, "train.csv"))

    # Apply backtranslate
    backtranslated_df = backtranslate_dataset(train)

    # Remove undesired special tokens
    backtranslated_df_cleaned = remove_special_tokens(backtranslated_df, 'text')

    # save to csv
    df = pd.DataFrame(backtranslated_df_cleaned)
    output_path = os.path.join(Path(defines.DATA_DIR), 'backtranslate_all.csv')
    df.to_csv(output_path, index=False)


def backtranslate_dataset(dataset):
    """
        This function backtranslates the text in the given dataset using multiple processes for faster execution.
        It utilizes Python's multiprocessing.Pool to parallelize the backtranslation process across multiple CPU cores.

        :param dataset: pd.DataFrame: A pandas DataFrame containing the dataset with a 'text' column to backtranslate.
        :return: list of dicts: backtranslated text and target labels for each tasks and the rewired ID.
    """
    with Pool(processes=6) as pool:
        backtranslated_dataset = list(tqdm(pool.imap(backtranslate_example, dataset.iterrows()),
                                           total=len(dataset),
                                           desc="Backtranslating"))
    return backtranslated_dataset

def backtranslate_example(row):
    """
        This function takes a row from a dataset and backtranslates the 'text' column using the
        backtranslate function. It then constructs a new dictionary containing the original
        'rewire_id', backtranslated 'text', and the original target labels ('target_a', 'target_b', and 'target_c').

        :param row: Tuple[int, pd.Series]: A tuple containing the index and the data of the row.
            The data should contain the columns 'rewire_id', 'text', 'target_a', 'target_b', and 'target_c'.
        :return: Dict[str, Union[int, str]]: A dictionary containing the 'rewire_id', backtranslated 'text',
            and the original target labels ('target_a', 'target_b', and 'target_c').
    """
    idx, data = row

    rewire_id = data['rewire_id']
    target_a = data['target_a']
    target_b = data['target_b']
    target_c = data['target_c']

    backtranslated_text = backtranslate(data['text'])

    backtranslated_review = {'rewire_id': rewire_id, 'text': backtranslated_text,
                             'target_a': target_a,
                             'target_b': target_b,
                             'target_c': target_c}
    return backtranslated_review

def backtranslate(text, source_lang='en', target_lang='fr'):
    try:
        # Initialize the tokenizer and model for translation to the target language
        tokenizer_to = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')
        model_to = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')

        # Tokenize the text and translate to the target language
        encoded = tokenizer_to.encode(text, return_tensors='pt')
        translated = model_to.generate(encoded)
        target_text = tokenizer_to.decode(translated[0])

        # Initialize the tokenizer and model for translation back to the original language
        tokenizer_back = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}')
        model_back = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}')

        # Tokenize the target language text and translate back to the original language
        encoded = tokenizer_back.encode(target_text, return_tensors='pt')
        back_translated = model_back.generate(encoded)
        backtranslated_text = tokenizer_back.decode(back_translated[0])

        return backtranslated_text

    except Exception as e:
        print(f"Error during backtranslation: {e}")
        print(f"Original text: {text}")
        return text


def remove_special_tokens(df, column):
    df[column] = df[column].apply(process_string)
    return df


def process_string(s):
    s = s.lstrip('<pad>').rstrip('</s>')
    return s.strip()


if __name__ == "__main__":
    main()