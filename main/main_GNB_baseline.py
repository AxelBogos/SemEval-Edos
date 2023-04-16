from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

from src.data.text_processing import TextPreprocessor
from src.utils import defines


def main() -> None:
    """The main function of this module. Runs a multinomial-GNB baseline on all tasks.

    :return: Nothing
    """

    # Load train
    train = pd.read_csv(Path(defines.INTERIM_DATA_DIR, "train.csv"))

    # Load val
    val = pd.read_csv(Path(defines.INTERIM_DATA_DIR, "val.csv"))
    # Load test
    test = pd.read_csv(Path(defines.INTERIM_DATA_DIR, "test.csv"))

    # Task A
    eval_task(train, val, test, "target_a", "A")
    # Task B
    eval_task(train, val, test, "target_b", "B")
    # Task C
    eval_task(train, val, test, "target_c", "C")


def eval_task(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_label: str, task: str
) -> None:
    """The eval_task function takes in the train, dev and test dataframes as well as the target_col
    label and task name. It then performs basic text processing on all three datasets (train, dev
    and test) using a TextProcessor object. The function then filters out any rows from the train
    dataset that have.

    a - 1 value for their target_col label (this is because we are only interested in training our model on
    the tasks that have been labelled). The function then applies a CountVectorizer to each of these datasets,
    fitting it to the training set but not transforming it. Finally, we fit a Multinomial-GNB.

    :param train: pd.DataFrame: Pass the train dataframe
    :param val: pd.DataFrame: Pass the dev dataframe to the function
    :param test: pd.DataFrame: Evaluate the model on the test set
    :param target_label: str: Filter the train data for the current task
    :param task: str: Print the task name in the output
    :return: None. Prints macro f-score for the validation and test set
    """

    # Basic tex processing (remove stopwords, urls, lowercasing etc..)
    text_processor = get_text_processor()

    # Filter data for the current task
    train = train[train[target_label] != -1]
    val = val[val[target_label] != -1]
    test = test[test[target_label] != -1]

    # Apply text processing and convert to np array
    X_train, y_train = np.array(text_processor.transform_series(train["text"])), np.array(
        train[target_label]
    )
    X_val, y_val = np.array(text_processor.transform_series(val["text"])), np.array(
        val[target_label]
    )
    X_test, y_test = np.array(text_processor.transform_series(test["text"])), np.array(
        test[target_label]
    )

    # Vectorize with counter Vectorizer
    count_vectorizer = CountVectorizer()
    X_train = count_vectorizer.fit_transform(X_train)
    X_val = count_vectorizer.transform(X_val)
    X_test = count_vectorizer.transform(X_test)

    # Model and predict
    clf = MultinomialNB()
    clf.fit(X=X_train, y=y_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    print(f"-------Task {task} baseline GNB--------")
    print(f'-->\t\t Val Macro F1: {f1_score(y_val, y_val_pred, average="macro"):.5f}')
    print(f'-->\t\t Test Macro F1: {f1_score(y_test, y_test_pred, average="macro"):.5f}\n')


def get_text_processor():
    """The get_text_processor function returns a TextPreprocessor object with the following
    attributes:

        - lemmatize = False
        - tokenize = False

    :return: A textpreprocessor object
    """
    text_processor = TextPreprocessor()
    text_processor.lemmatize = False
    text_processor.tokenize = False
    return text_processor


if __name__ == "__main__":
    main()
