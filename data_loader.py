import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import torch_geometric.utils as gm_utils

import src.Preprocessing as dpp
import networkx as nx
import scipy.sparse as sp
from torch_geometric.datasets import KarateClub
from tqdm.std import tqdm
from torch_geometric.datasets import Planetoid


def load_bail():
    data_path_root = ''
    adj, features, labels, _, _, _, _, _, _ = dpp.load_data(data_path_root, "bail")

    # get first n_nodes nodes
    n_nodes = 500
    adj = adj[:n_nodes, :n_nodes]
    adj = adj - sp.eye(adj.shape[0])
    features = features[:n_nodes]
    labels = labels[:n_nodes]

    # create pyg graph
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    data = Data(x=features, edge_index=edge_index)
    data.y = labels

    # Create masks
    num_nodes = features.shape[0]
    indices = torch.randperm(num_nodes)
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True

    # Assign masks to the data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data

# def feature_selection(features):
#     return 


def attr_tag(data, setting):
    if setting == 'sa':
        node_feature_strings = [''.join(map(str, features.tolist())) for features in data.x]
        # Map combined attribute strings to unique integers
        string_to_int_map = {string: i for i, string in enumerate(set(node_feature_strings))}
        # Convert combined attribute strings to their corresponding unique integers
        node_labels = torch.tensor([string_to_int_map[string] for string in node_feature_strings], dtype=torch.long)
        data.wl_y = node_labels
    elif setting == 'ss':
        node_feature_strings = [''.join(map(str, features.tolist()[:3])) for features in data.x]
        string_to_int_map = {string: i for i, string in enumerate(set(node_feature_strings))}
        node_labels = torch.tensor([string_to_int_map[string] for string in node_feature_strings], dtype=torch.long)
        data.wl_y = node_labels
    elif setting == 's':
        data.wl_y = torch.ones(data.y.size(0), dtype=torch.long)
    return data

def dataset_func(dataset_name):

    setting = 'sa' # s, ss, sa / structure, structure subset, structure all

    if dataset_name == 'bail':
        data = load_bail()
        data = attr_tag(data, setting)
        return data
    if dataset_name == 'cora':
        data = Planetoid(root='/tmp/Cora', name='Cora')[0]
        data = attr_tag(data, setting)
        return data
