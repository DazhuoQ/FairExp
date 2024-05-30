from tqdm.std import tqdm
import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from data_loader import dataset_func
import numpy as np
from wl_similarity import sim_wl
import torch.nn.functional as F
from model import GCN
from LSH import EuclideanLSH


def extract_k_hop_subgraphs(node, graph, num_hops=3):
        
    # Extract the 3-hop subgraph surrounding the test node
    subgraph_node_indices, subgraph_edge_indices, mapping, edge_mask = k_hop_subgraph(
        node_idx=node.item(), num_hops=num_hops, edge_index=graph.edge_index, relabel_nodes=True)
    
    # Prepare the subgraph data
    subgraph_features = graph.x[subgraph_node_indices]
    subgraph_labels = graph.y[subgraph_node_indices]
    wl_labels = graph.wl_y[subgraph_node_indices]
    subgraph_edge_index = subgraph_edge_indices

    # Create a new Data object for the subgraph
    subgraph = Data(x=subgraph_features, edge_index=subgraph_edge_index, y=subgraph_labels, wl_y=wl_labels)
    
    return subgraph


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


if __name__ == "__main__":

    k = 5
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_name = 'cora'

    graph = dataset_func(dataset_name)
    datax = np.array(graph.x[:,:100])
    print('dataset prepared. ')
    lsh = EuclideanLSH(10, 1, 100)
    lsh.insert(datax)
    
    index = np.random.randint(0, datax.shape[0])
    query = datax[int(index)]
    
    res = lsh.query(query, 20)
    print(np.sqrt(np.sum(np.power(res - query, 2), axis=-1)))
    print("-"*20)
    sort = np.argsort(np.sum(np.power(datax - query, 2), axis=-1))
    print(np.sqrt(np.sum(np.power(datax[sort[:20]] - query, 2), axis=-1)))
    print("-"*20)
    print(np.sqrt(np.sum(np.power(datax[sort[-20:]] - query, 2), axis=-1)))