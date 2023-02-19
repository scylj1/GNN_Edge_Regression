#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:rtx2080:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=dgb # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate dgb

data=UNtrade
n_runs=5

# TGN
method=tgn
prefix="${method}_attn"
python tgn/train_tgn_regression.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0
