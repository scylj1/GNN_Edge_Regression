This reporsitory is for the L45: Representation Learning on Graphs and Networks. 

---

## Running the experiments

### Set up Environment
```{bash}
conda create -n dgb python=3.9
```

then run 
```{bash}
source install.sh
```

### The basic scripts for each model is as follows:

```{bash}
data=UNtrade
n_runs=1

# TGN
method=tgn
prefix="${method}_attn"
python tgn/train_self_supervised.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0

# JODIE
method=jodie
prefix="${method}_rnn"
python tgn/train_tgn_regression.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0

# DyRep
method=dyrep
prefix="${method}_rnn"
python tgn/train_tgn_regression.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0

```

### The following normalization methods are available:
```{bash}
# Just choose one of them is fine!

--max_normalization

--logarithmize_weights

--node_out_normalization

(--node_in_normalization)
```

The following sampling methods are available:
```{bash}
# TODO
```