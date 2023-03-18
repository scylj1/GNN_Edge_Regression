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
    parser.add_argument('--seeds', type=int, default=0, help='seeds')
    parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--max_normalization', action='store_true',
                        help='Whether use min max normalization on weights')
    parser.add_argument('--logarithmize_weights', action='store_true',
                        help='Whether to logarithmize weights')

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
                              max_normalization=args.max_normalization,
                              logarithmize_weights=args.logarithmize_weights)

    train_data = wrap_data(train_x[0], edge_index_source, edge_index_dest, train_edge.reshape((-1, train_edge.shape[-1])))
    val_data = wrap_data(val_x[0], edge_index_source, edge_index_dest, val_edge.reshape((-1, val_edge.shape[-1])))
    test_data = wrap_data(test_x[0], edge_index_source, edge_index_dest, test_edge.reshape((-1, test_edge.shape[-1])))
    
    test_edge_pos = test_edge.reshape((-1, test_edge.shape[-1]))
    
    neg_index = []
    for i in range(len(test_edge_pos)):
        if all(element == 0 for element in test_edge_pos[i]):
            neg_index.append(i)
  
    test_edge_pos = np.delete(test_edge_pos, neg_index, axis=0)
    edge_index_source_pos = np.delete(edge_index_source, neg_index, axis=0)
    edge_index_dest_pos = np.delete(edge_index_dest, neg_index, axis=0)
    
    test_data_pos = wrap_data(test_x[0], edge_index_source_pos, edge_index_dest_pos, test_edge_pos)
    

    return train_data.to(device), val_data.to(device), test_data.to(device), test_data_pos.to(device)

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
    args = parse_args()
    
    torch.manual_seed(args.seeds)
    np.random.seed(args.seeds)  
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
   
    get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data processing
    train_data, val_data, test_data, test_data_pos = process_data(args, device)
    train_loader = DataLoader([train_data], batch_size=args.bs, shuffle=True)

    # model initilisation
    model = EdgeRegGNN(num_node_features=train_data.num_node_features, hidden_channels=64, num_edge_features=train_data.edge_attr.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False, tolerance=1e-20)
    
    # train
    epochs = args.n_epoch
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
        
        pred = model(test_data_pos)
        loss = edge_regression_loss(pred, test_data_pos.edge_attr)
        print(f"Test loss postive: {loss:.4f}")
    

if  __name__=="__main__":
    main()
