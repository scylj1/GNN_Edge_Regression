#!/bin/bash

data=UNtrade
n_runs=3

# Regression

#baseline
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs 1 --gpu 0 --logarithmize_weights --n_epoch 1 --do_baseline
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs 1 --gpu 0 --max_normalization --n_epoch 1 --do_baseline
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs 1 --gpu 0 --node_out_normalization --n_epoch 1 --do_baseline


# jodie
#method=jodie
#prefix="${method}_rnn"

#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200 
#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200 
#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200 

#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200 --no_negative_sampling
#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200 --no_negative_sampling
#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200 --no_negative_sampling

#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200 --no_negative_sampling --fill_all_edges
#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200 --no_negative_sampling --fill_all_edges
#python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200 --no_negative_sampling --fill_all_edges

# DyRep
#method=dyrep
#prefix="${method}_rnn"

#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200 
#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200 
#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200 

#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200 --no_negative_sampling 
#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200 --no_negative_sampling
#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200 --no_negative_sampling

#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200 --no_negative_sampling --fill_all_edges
#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200 --no_negative_sampling --fill_all_edges
#python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200 --no_negative_sampling --fill_all_edges

# TGN
#method=tgn
#prefix="${method}_attn"
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200

#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200 --no_negative_sampling
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200 --no_negative_sampling
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200 --no_negative_sampling

#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --logarithmize_weights --n_epoch 200 --no_negative_sampling --fill_all_edges
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --max_normalization --n_epoch 200 --no_negative_sampling --fill_all_edges
#python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --node_out_normalization --n_epoch 200 --no_negative_sampling --fill_all_edges

# GCN
#python gcn/train_gcn_regression.py -d "UNtrade" --logarithmize_weights
#python gcn/train_gcn_regression.py -d "UNtrade" --max_normalization
#python gcn/train_gcn_regression.py -d "UNtrade" --node_out_normalization


# Classification 
#baseline 
#python tgn/train_tgn_classification.py -d $data --use_memory --prefix "$prefix" --n_runs 1 --gpu 0 --n_epoch 1 --num_class 10

# TGN
#method=tgn
#prefix="${method}_attn"
#python tgn/train_tgn_classification.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --n_epoch 200 --num_class 10 
#python tgn/train_tgn_classification.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --n_epoch 200 --num_class 10 --no_negative_sampling

#method=jodie
#prefix="${method}_rnn"
#python tgn/train_tgn_classification.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --n_epoch 200 --num_class 10
#python tgn/train_tgn_classification.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --n_epoch 200 --num_class 10 --no_negative_sampling

#method=dyrep
#prefix="${method}_rnn"
#python tgn/train_tgn_classification.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --n_epoch 200 --num_class 10
#python tgn/train_tgn_classification.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0 --n_epoch 200 --num_class 10 --no_negative_sampling

