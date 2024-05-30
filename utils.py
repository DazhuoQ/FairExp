import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import src.Preprocessing as dpp
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid


def load_bail():
    data_path_root = ''
    adj, features, labels, _, _, _, _, _, _ = dpp.load_data(data_path_root, "bail")

    # get first n_nodes nodes
    n_nodes = 2000
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
    train_size = int(0.3 * num_nodes)
    val_size = int(0.1 * num_nodes)

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
    elif setting == 'f':
        data.wl_y = torch.ones(data.y.size(0), dtype=torch.long)
    return data


def feature_selection(data, setting):
    if setting == 'sa':
        data.wl_x = data.x
        return data
    elif setting == 'ss':
        num_features = data.x.size(1)
        num_nodes = data.x.size(0)
        quarter_idx = num_features // 4
        masked_x = torch.cat([data.x[:, :quarter_idx], torch.zeros((num_nodes, num_features - quarter_idx))], dim=1)
        data.wl_x = masked_x
        return data
    elif setting == 's':
        data.wl_x = torch.ones((data.x.size(0), 2))
    elif setting == 'f':
        data.wl_x = data.x
        return data
    return data


def dataset_func(dataset_name, setting):

    # setting = 'sa' # s, ss, sa / structure, structure subset, structure all

    if dataset_name == 'bail':
        data = load_bail()
        data = attr_tag(data, setting)
        data = feature_selection(data, setting)
        return data
    if dataset_name == 'cora':
        data = Planetoid(root='/tmp/Cora', name='Cora')[0]
        data = attr_tag(data, setting)
        data = feature_selection(data, setting)
        return data


def extract_k_hop_subgraphs(node, graph, num_hops=3):
        
    # Extract the 3-hop subgraph surrounding the test node
    subgraph_node_indices, subgraph_edge_indices, mapping, edge_mask = k_hop_subgraph(
        node_idx=node.item(), num_hops=num_hops, edge_index=graph.edge_index, relabel_nodes=True)
    
    # Prepare the subgraph data
    subgraph_features = graph.x[subgraph_node_indices]
    subgraph_labels = graph.y[subgraph_node_indices]
    wl_labels = graph.wl_y[subgraph_node_indices]
    wl_x = graph.wl_x[subgraph_node_indices]
    subgraph_edge_index = subgraph_edge_indices

    # Create a new Data object for the subgraph
    subgraph = Data(x=subgraph_features, edge_index=subgraph_edge_index, y=subgraph_labels, wl_y=wl_labels, wl_x=wl_x)
    
    return subgraph, mapping


def generate_unique_pairs(arr):
    unique_pairs = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            unique_pairs.add((arr[i], arr[j]))
    return unique_pairs


def clean_data(test_nodes, graph, node1):
    all_nodes = set(range(graph.num_nodes))
    connected_nodes = set(graph.edge_index.view(-1).tolist())
    isolated_nodes = torch.tensor(sorted(list(all_nodes - connected_nodes)))
    mask = ~test_nodes.unsqueeze(1).eq(isolated_nodes).any(1)
    clean_test_nodes = test_nodes[mask]
    mask = clean_test_nodes != node1
    clean_test_nodes = clean_test_nodes[mask]
    return clean_test_nodes
