import math
import logging
import time
import datetime
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction_modified
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from collections import Counter
import pandas as pd

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
# additional arguments
parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of the validation data.')
parser.add_argument('--test_ratio', type=float, default=0.15, help="Ratio of the test data.")


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)
  
DATA = args.data
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA, args.val_ratio, args.test_ratio,
                              different_new_nodes_between_val_and_test=args.different_new_nodes,
                              randomize_features=args.randomize_features)

#print(len(edge_features))
#print(len(node_features))
#print(node_features)

#print(train_data.edge_idxs)
#print(val_data.edge_idxs)
#print(test_data.edge_idxs)

#print(train_data.sources)
#print(train_data.destinations)
#print(train_data.timestamps)
#print(len(train_data.edge_idxs))
#print(edge_features)
  
# find node with value
indexes = []
for index, source in enumerate(train_data.sources):
  if source == 56 and train_data.destinations[index] == 127:
    indexes.append(index)   
#print(indexes)


timesta = []
values = []
for index in indexes:
  #print(edge_features[train_data.edge_idxs[index]])
  #print(train_data.timestamps[index])
  timesta.append(train_data.timestamps[index])
  values.append(edge_features[train_data.edge_idxs[index]])


distinct_list= (Counter(train_data.timestamps).keys())
#print("List with distinct elements:\n",len(distinct_list))
data_len = len(distinct_list)
edge_values = []
times = []
data = pd.DataFrame()

for i, name in enumerate(distinct_list): 
  #print(int(int(i)/31536000))
  times.append(name)
  for index, t in enumerate(timesta):
    if t == name:
      edge_value = int(values[index])
      edge_values.append(int(values[index]))
      
  if len(edge_values) != len(times):
    edge_value = 0
    edge_values.append(0)
    
  data = data.append(pd.DataFrame({'value':[edge_value],'time':[name]}),ignore_index=True)

#print(edge_values)
#print(times)


import decimal
import sys
import os
import statsmodels
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings
import seaborn as sns
import matplotlib as matplotlib
import math
import matplotlib
#from keras.losses import mean_squared_error, mean_absolute_error
from pasta.augment import inline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from arch.unitroot import ADF
import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import style
style.use('ggplot')
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.api import qqplot
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=5, suppress=True)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
pd.set_option('display.width', 10000)

print(data['value'])

print("原始单位根检验:\n")
print(ADF(data['value'])) #原始

data["diff1"] = data["value"].diff(3).dropna()
print("一阶单位根检验:\n")
print(ADF(data.diff1.dropna())) #一阶

    