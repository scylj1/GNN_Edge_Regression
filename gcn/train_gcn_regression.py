import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import argparse
import logging
from pathlib import Path
from utils.data_processing import get_data, EarlyStopMonitor

def parse_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('GCN training')
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
    parser.add_argument('--max_normalization', action='store_true',
                        help='Whether use min max normalization on weights')
    parser.add_argument('--logarithmize_weights', action='store_true',
                        help='Whether to logarithmize weights')
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
    
    return args
  
class EdgeRegGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_edge_features):
        super(EdgeRegGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(2 * hidden_channels, num_edge_features)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        x_i = torch.index_select(x, 0, edge_index[0])
        x_j = torch.index_select(x, 0, edge_index[1])
        out = self.lin(torch.cat([x_i, x_j], dim=-1))

        return out

# loss function
def edge_regression_loss(pred, target):
    return F.mse_loss(pred, target)

# data processing
def process_data(args, device):

    train_x, val_x, test_x, train_edge, val_edge, test_edge, edge_index_source, edge_index_dest = get_data(args.data, args.val_ratio, args.test_ratio,
                              different_new_nodes_between_val_and_test=args.different_new_nodes,
                              randomize_features=args.randomize_features, 
                              max_normalization=args.max_normalization,
                              logarithmize_weights=args.logarithmize_weights)

    train_data = wrap_data(train_x[0], edge_index_source, edge_index_dest, train_edge.reshape((-1, train_edge.shape[-1])))
    val_data = wrap_data(val_x[0], edge_index_source, edge_index_dest, val_edge.reshape((-1, val_edge.shape[-1])))
    test_data = wrap_data(test_x[0], edge_index_source, edge_index_dest, test_edge.reshape((-1, test_edge.shape[-1])))

    return train_data.to(device), val_data.to(device), test_data.to(device)

def wrap_data(x, edge_source, edge_dest, edge_attr):
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor([edge_source, edge_dest], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)   
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# train
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = edge_regression_loss(out, data.edge_attr)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    torch.manual_seed(0)
    np.random.seed(0)  
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    # MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
    get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data processing
    train_data, val_data, test_data = process_data(args, device)
    train_loader = DataLoader([train_data], batch_size=64, shuffle=True)

    # model initilisation
    model = EdgeRegGNN(num_node_features=train_data.num_node_features, hidden_channels=64, num_edge_features=train_data.edge_attr.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False, tolerance=1e-20)
    
    # train
    epochs = 200
    for epoch in range(epochs):
        for data_item in train_loader:
            loss = train(model, data_item, optimizer)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {loss:.4f}")
            
        # evaluate
        model.eval()
        with torch.no_grad():
            pred = model(val_data)
            loss = edge_regression_loss(pred, val_data.edge_attr)
            print(f"Eval loss: {loss:.4f}")
        
        if early_stopper.early_stop_check(loss):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), get_checkpoint_path(epoch))
    
    with torch.no_grad():
        pred = model(test_data)
        loss = edge_regression_loss(pred, test_data.edge_attr)
        print(f"Test loss: {loss:.4f}")
    

if  __name__=="__main__":
    main()
