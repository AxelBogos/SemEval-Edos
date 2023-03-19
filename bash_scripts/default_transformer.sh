#!/bin/bash
# pip install -r requirements.txt
python ./../main/main_make_dataset.py
python ./../main/main.py \
--train \
--eval \
--task a \
--preprocessing_mode none \
--model roberta-base \
--optimizer AdamW \
--max_token_length 128 \
--step_scheduler  5 \
--n_warmup_steps 0 \
--lr 5e-6 \
--num_epoch 9 \
--batch_size 16 \
--patience 3 \
