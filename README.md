<div align="center">

# Explainable Detection of Online Sexism (EDOS)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
[![Repo](https://img.shields.io/badge/GitHub-Repo-lightgrey)](https://github.com/AxelBogos/SemEval-Edos)
[![Conference](https://img.shields.io/badge/WandB-Experiments-yellow)](https://wandb.ai/axel-bogos/EDOS-ift6289?workspace=)

</div>

## Project Organization

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
│
├── main
│   ├── main.py                     <- Main entry point for per-level transformer & lstm models.
│   ├── main_GNB_baseline           <- Main entry point for GNB baselines runs.
│   ├── main_hierichal              <- Main entry point for hierarichal masking models.
│   └── main_make_dataset           <- Main entry point for downloading light processing of dataset.
│   └── main_parent_conditioning    <- Main entry point for parent conditioned-models.
│
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   └── raw            <- The original, immutable data dump.
│
├── logs               <- Runs logs and serialized models
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── app.py         <- Streamlit app
│   │
│   ├── data           <- lightning data modules
│   │
│   ├── models         <- lightning modules
│   │
│   ├── utils         <- helper functions and modules
│   │
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

______________________________________________________________________

## Abstract

In this technical report, we present a comprehensive study on the identification and classification of sexist language on online social platforms, conducted in the context of SemEval 2023 Task 10. This task is particularly relevant in the current context of growing concerns about hateful speech on certain social media platforms. The task organizers provide a data set of 20,000 labeled entries from Gab and Reddit, which is divided into three hierarchical classification tasks. We employ various machine learning models, including Gaussian Naive Bayes, bi-LSTM, and Transformer-based models, to perform classification tasks and explore a variety of hierarchical classification methodologies. Our best results for each task are obtained using the RoBERTa-large model and are ranked within the top 10% of entries on the competition leaderboard for subtask B. We then outline future work, which includes better leveraging the hierarchical taxonomy in the classification architecture and enhancing explainability in a generative setting.

## How to run

Clone Repo

```bash
git clone https://github.com/AxelBogos/SemEval-Edos.git
cd SemEval-Edos/
```

Install dependencies

```bash
pip install -r requirements.txt -q
```

Download Dataset (~5MB main, 2GB with external data)

```bash
python ./main/main_make_dataset.py
```

Have fun with the basic transformer model on Task A

```bash
python ./main/main.py
        --train \
        --eval \
        --task a \
        --preprocessing_mode none \
        --architecture transformer \
        --model distilroberta-base \
        --lr 1e-05 \
        --num_epoch 9 \
        --batch_size 16 \
        --patience 3 \
```
