from tqdm.std import tqdm
import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from data_loader import dataset_func
import numpy as np
from wl_similarity import sim_wl
import torch.nn.functional as F
from model import GCN


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
    dataset_name = 'bail'

    graph = dataset_func(dataset_name)
    print('dataset prepared. ')

    # Extract 3-hop subgraphs for one test node
    test_nodes = torch.where(graph.test_mask)[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GCN(input_dim=graph.x.size(1), hidden_dim=16, output_dim=graph.y.size(0))
    model.load_state_dict(torch.load('models/{}_gcn_model.pth'.format(dataset_name)))
    model.eval()
    model.to(device)

    out = model(graph)
    probs = F.softmax(out, dim=1)

    # given a node
    n_idx = seed
    node1 = test_nodes[n_idx]
    pred_label_1 = probs[node1].argmax().item()
    subgraph1 = extract_k_hop_subgraphs(node1, graph)
    remaining_nodes = clean_data(test_nodes, graph, node1)

    rank_dict = {}
    for node2 in tqdm(remaining_nodes, desc='num_nodes'):
        pred_label_2 = probs[node2].argmax().item()
        subgraph2 = extract_k_hop_subgraphs(node2, graph)
        sim_score = sim_wl(subgraph1, subgraph2, 3).item()
        # if sim_score > 0.1:
        if pred_label_1 != pred_label_2:
            print(f'counterfactual: {pred_label_1}, {pred_label_2}')
            print(f'real counterfactual: {graph.y[node1]}, {graph.y[node2]}')
            rank_dict[node2.item()] = sim_score
    # top_k = sorted(rank_dict.items())[:k]
    top_k = sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)[:k]

    print(top_k)