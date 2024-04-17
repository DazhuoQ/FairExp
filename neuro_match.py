from tqdm.std import tqdm
import argparse
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from data_loader import dataset_func
import numpy as np

import networkx as nx
from wl_similarity import sim_wl
from neural_subgraph_learning.common import utils
from neural_subgraph_learning.subgraph_matching.config import parse_encoder
from neural_subgraph_learning.subgraph_matching.train import build_model

def extract_k_hop_subgraphs(node, graph, num_hops=3):
        
    # Extract the 3-hop subgraph surrounding the test node
    subgraph_node_indices, subgraph_edge_indices, mapping, edge_mask = k_hop_subgraph(
        node_idx=node, num_hops=num_hops, edge_index=graph.edge_index, relabel_nodes=True)
    
    # Prepare the subgraph data
    subgraph_features = graph.x[subgraph_node_indices]
    subgraph_labels = graph.y[subgraph_node_indices]
    wl_labels = graph.wl_y[subgraph_node_indices]
    subgraph_edge_index = subgraph_edge_indices

    # Create a new Data object for the subgraph
    subgraph = Data(x=subgraph_features, edge_index=subgraph_edge_index, y=subgraph_labels, wl_y=wl_labels)
    
    out = to_networkx(subgraph,to_undirected=True)
    return out, mapping.item()


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

def gen_alignment_matrix(model, query, target, u, v, method_type="order"):
    """Generate subgraph matching alignment matrix for a given query and
    target graph. Each entry (u, v) of the matrix contains the confidence score
    the model gives for the query graph, anchored at u, being a subgraph of the
    target graph, anchored at v.

    Args:
        model: the subgraph matching model. Must have been trained with
            node anchored setting (--node_anchored, default)
        query: the query graph (networkx Graph)
        target: the target graph (networkx Graph)
        method_type: the method used for the model.
            "order" for order embedding or "mlp" for MLP model
    """
    batch = utils.batch_nx_graphs([query, target], anchors=[u, v])
    embs = model.emb_model(batch)
    pred = model(embs[1].unsqueeze(0), embs[0].unsqueeze(0))
    raw_pred = model.predict(pred)

    if method_type == "order":
        raw_pred = torch.log(raw_pred)
    elif method_type == "mlp":
        raw_pred = raw_pred[0][1]

    return raw_pred.item()



if __name__ == "__main__":

    k = 5
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    graph = dataset_func('bail')
    print('dataset prepared. ')

    # Extract 3-hop subgraphs for one test node
    test_nodes = torch.where(graph.test_mask)[0]

    # given a node
    n_idx = seed
    node1 = test_nodes[12]
    subgraph1, mapping1 = extract_k_hop_subgraphs(node1.item(), graph)
    nx.draw(subgraph1)
    plt.show()
    remaining_nodes = clean_data(test_nodes, graph, node1)
    
    
    parser = argparse.ArgumentParser(description='Alignment arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument('--query_path', type=str, help='path of query graph',
        default="")
    parser.add_argument('--target_path', type=str, help='path of target graph',
        default="")
    args = parser.parse_args()
    args.test = True
    model = build_model(args)
    

    rank_dict = {}
    for node2 in remaining_nodes:
        subgraph2, mapping2 = extract_k_hop_subgraphs(node2.item(), graph)
        sim_score = gen_alignment_matrix(model, subgraph1, subgraph2, mapping1, mapping2)
        print(sim_score)
        # if sim_score > 0.1:
        rank_dict[node2.item()] = sim_score
    # top_k = sorted(rank_dict.items())[:k]
    top_k = sorted(rank_dict.items(), key=lambda item: item[1])[:k]
    
    for node, score in top_k:
        print(f"node:{node},score:{score}")
        subgraph, _ = extract_k_hop_subgraphs(node, graph)
        nx.draw(subgraph)
        plt.show()

    print(top_k)