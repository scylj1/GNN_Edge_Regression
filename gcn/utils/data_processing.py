import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_data(dataset_name, val_ratio, test_ratio, max_normalization=False, logarithmize_weights=False, node_out_normalization = False, node_in_normalization = False):
    ### Load data
    dataset_name = "UNtrade"
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    edge_features = edge_features[1:]
    
    # normalisation
    if max_normalization:
        scaler = MinMaxScaler(feature_range=(0, 10))
        edge_features = scaler.fit_transform(edge_features)

    # logarithmize weights
    if logarithmize_weights:
        edge_features = np.log10(edge_features)
        # if after logarithm, the weight is too low, we set it to 0.001.
        edge_features = np.maximum(edge_features, 0.001)
    
    # for a given source node, divide all edges weights by the max edge weight that node has in a timestamp
    if node_out_normalization:
        print("Using node_out_normalization...")
        graph_df['weight'] = edge_features
        unique_timestamps = graph_df.ts.unique()
        for t in unique_timestamps:
            unique_source = graph_df[graph_df.ts == t].u.unique()
            for x in unique_source:
                edges = graph_df[(graph_df.u == x) & (graph_df.ts == t)]
                # Calculate the max weight for these edges
                weight_sum = edges['weight'].sum()
                if weight_sum!=0:
                    # Divide all edge weights by the max weight
                    graph_df.loc[(graph_df.u == x) & (graph_df.ts == t), 'weight'] = graph_df['weight'] / weight_sum
        edge_features = graph_df.weight.values
        edge_features = edge_features.reshape(-1, 1)

    # for a given destination node, divide all edges weights by the max edge weight that node has in a timestamp
    if node_in_normalization:
        print("Using node_in_normalization...")
        graph_df['weight'] = edge_features
        unique_timestamps = graph_df.ts.unique()
        for t in unique_timestamps:
            unique_des = graph_df[graph_df.ts == t].i.unique()
            for x in unique_des:
                edges = graph_df[(graph_df.i == x) & (graph_df.ts == t)]
                # Calculate the max weight for these edges
                weight_sum = edges['weight'].sum()
                if weight_sum!=0:
                    # Divide all edge weights by the max weight
                    graph_df.loc[(graph_df.i == x) & (graph_df.ts == t), 'weight'] = graph_df['weight'] / weight_sum
        edge_features = graph_df.weight.values
        edge_features = edge_features.reshape(-1, 1)
    
    timestamps = [int(timestamps[i] / 31536000) for i in range(len(timestamps))]
    unique_values, index = np.unique(timestamps, return_index=True)
    unique_times = unique_values[np.argsort(index)]

    n_unique_times = len(unique_times)
    
    unique_nodes = set(sources) | set(destinations)
    n_unique_nodes = len(unique_nodes)
    
    edge_index_source = np.arange(n_unique_nodes).repeat(n_unique_nodes)
    edge_index_dest = np.arange(n_unique_nodes)
    edge_index_dest = np.tile(edge_index_dest, n_unique_nodes)
    
    edge_feature_matrix = np.zeros((n_unique_nodes, n_unique_nodes, n_unique_times)) 
    for i in range(len(sources)):
        edge_feature_row = edge_feature_matrix[int(sources[i]-1)]
        edge_feature = edge_feature_row[int(destinations[i]-1)]
        edge_feature[int(timestamps[i])] = float(edge_features[i])
    
    x_matrix = np.zeros((n_unique_nodes, n_unique_nodes, n_unique_times))
    x_matrix[:][:] = unique_times
    
    val_ratio, test_ratio = 0.15, 0.15
    test_len = int(n_unique_times * test_ratio)
    val_len = int(n_unique_times * val_ratio)
    train_len = n_unique_times - test_len - val_len
    
    train_x_matrix = x_matrix[:, :, 0:train_len]
    val_x_matrix = x_matrix[:, :, train_len:(train_len + val_len)]
    test_x_matrix = x_matrix[:, :, (train_len + val_len):]
    
    train_edge_matrix = edge_feature_matrix[:, :, 0:train_len]
    val_edge_matrix = edge_feature_matrix[:, :, train_len:(train_len + val_len)]
    test_edge_matrix = edge_feature_matrix[:, :, (train_len + val_len):]
    
    train_x_matrix_modified = np.zeros((n_unique_nodes, n_unique_nodes, test_len))
    val_x_matrix_modified = np.zeros((n_unique_nodes, n_unique_nodes, test_len))
    train_edge_matrix_modified = np.zeros((n_unique_nodes, n_unique_nodes, test_len))
    val_edge_matrix_modified = np.zeros((n_unique_nodes, n_unique_nodes, test_len))
    
    for i in range (n_unique_nodes):
        for j in range (n_unique_nodes):
            arr_split = np.split(train_x_matrix[i][j], test_len)
            arr_mean = np.array([np.mean(sub_arr) for sub_arr in arr_split])
            train_x_matrix_modified[i][j] = arr_mean
            
            arr_split = np.split(val_x_matrix[i][j], test_len)
            arr_mean = np.array([np.mean(sub_arr) for sub_arr in arr_split])
            val_x_matrix_modified[i][j] = arr_mean
            
            arr_split = np.split(val_edge_matrix[i][j], test_len)
            arr_mean = np.array([np.mean(sub_arr) for sub_arr in arr_split])
            val_edge_matrix_modified[i][j] = arr_mean
            
            arr_split = np.split(train_edge_matrix[i][j], test_len)
            arr_mean = np.array([np.mean(sub_arr) for sub_arr in arr_split])
            train_edge_matrix_modified[i][j] = arr_mean

    return train_x_matrix_modified, val_x_matrix_modified, test_x_matrix, train_edge_matrix_modified, val_edge_matrix_modified, test_edge_matrix, edge_index_source, edge_index_dest

class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round