import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from src.data.text_processing import TextPreprocessor
from src.utils import defines


def main() -> None:
    """The main function of this module. Runs a multinomial-GNB baseline on all tasks.
    :return: Nothing
    """

    # Load train
    train = pd.read_csv(Path(defines.INTERIM_DATA_DIR, "train.csv"))
    columns_train = train.columns

    # Apply backtranslate
    backtranslated_df = backtranslate_dataset(train[:5])

    # save to csv
    df = pd.DataFrame(backtranslated_df)
    output_path = os.path.join(Path(defines.DATA_DIR), 'backtranslate_all.csv')
    df.to_csv(output_path, index=False)


# Create a function to backtranslate an entire dataset
def backtranslate_dataset(dataset):

    backtranslated_dataset = []

    for idx in tqdm(range(len(dataset)), desc="Backtranslating"):
        # Get the targets

        rewire_id = dataset.iloc[idx]['rewire_id']
        target_a = dataset.iloc[idx]['target_a']
        target_b = dataset.iloc[idx]['target_b']
        target_c = dataset.iloc[idx]['target_c']

        # Backtranslate the text
        backtranslated_text = backtranslate(dataset.iloc[idx]['text'])

        # Create a new review with the backtranslated text and the same label
        backtranslated_review = {'rewire_id': rewire_id, 'text': backtranslated_text, 'target_a': target_a,
                                 'target_b': target_b, 'target_c': target_c}

        # Add the backtranslated review to the list
        backtranslated_dataset.append(backtranslated_review)

    return backtranslated_dataset


def backtranslate(text, source_lang='en', target_lang='fr'):
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


if __name__ == "__main__":
    main()