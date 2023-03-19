#!/bin/bash
# pip install -r requirements.txt
python ./../main/main_make_dataset.py
python ./../main/main.py \
--train \
--eval \
--task a \
--preprocessing_mode standard \
--model bilstm \
--lr 5e-6 \
--optimizer AdamW \
--embedding_dim 300 \
--hidden_dim 256 \
--num_layers 2 \
--bidirectional True \
--num_epoch 9 \
--batch_size 16 \
--patience 3 \
