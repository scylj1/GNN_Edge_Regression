#!/bin/bash

data=UNtrade
n_runs=1

# TGN
method=tgn
prefix="${method}_attn"
python tgn/train_tgn_classification.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --n_epoch 30 --num_class 3
