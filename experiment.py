import torch
import numpy as np
from relaxed_wl_similarity import aggregate_features, relaxed_sim_wl
from model import GCN
import torch.nn.functional as F
from tqdm.std import tqdm
from utils import *
from index_algorithm import *
from gin_algorithm import gin_emb, train_gin_model
from gat_algorithm import gat_emb, train_gat_model

if __name__ == "__main__":

    # Parameters
    k = 500
    seed = 12
    gamma = 0.1
    iterations = 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_name = 'cora'
    setting = 'sa'
    method = 'gin' # baseline index gin gat

    graph = dataset_func(dataset_name, setting)
    print('dataset prepared. ')

    test_nodes = torch.where(graph.test_mask)[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GCN(input_dim=graph.x.size(1), hidden_dim=16, output_dim=graph.y.size(0))
    model.load_state_dict(torch.load('models/{}_gcn_model.pth'.format(dataset_name)))
    model.eval()
    model.to(device)

    out = model(graph)
    probs = F.softmax(out, dim=1)

    # given a node
    node1 = test_nodes[seed]
    pred_label_1 = probs[node1].argmax().item()
    subgraph1, new_n1 = extract_k_hop_subgraphs(node1, graph)
    aggregate_features(subgraph1, iterations, gamma)
    remaining_nodes = clean_data(test_nodes, graph, node1)

    if method == 'index':
        candidates_idx = LSH_index(remaining_nodes, iterations, graph, node1, setting, seed, k)
    elif method == 'baseline':
        candidates_idx = remaining_nodes
    elif method == 'gin':
        candidates_idx = remaining_nodes
        train_gin_model(graph, seed)
    elif method == 'gat':
        candidates_idx = remaining_nodes
        train_gat_model(graph, seed)

    rank_dict = {}
    for node2 in tqdm(candidates_idx, desc='num_nodes'):
        pred_label_2 = probs[node2].argmax().item()
        subgraph2, new_n2 = extract_k_hop_subgraphs(node2, graph)
        aggregate_features(subgraph2, iterations, gamma)
        # sim_score = sim_wl(subgraph1, subgraph2, iterations).item()
        if setting == 'f':
            sim_score = F.cosine_similarity(graph.x[node1].unsqueeze(0), graph.x[node2].unsqueeze(0))
        else:
            if method == 'gin':
                emb = gin_emb()
                sim_score = F.cosine_similarity(emb[node1].unsqueeze(0), emb[node2].unsqueeze(0))
            elif method == 'gat':
                emb = gat_emb()
                sim_score = F.cosine_similarity(emb[node1].unsqueeze(0), emb[node2].unsqueeze(0))
            else:
                sim_score = relaxed_sim_wl(subgraph1, subgraph2, new_n1, new_n2)
        # if sim_score > 0.1:
        if pred_label_1 != pred_label_2:
            # print(f'counterfactual: {pred_label_1}, {pred_label_2}')
            # print(f'real counterfactual: {graph.y[node1]}, {graph.y[node2]}')
            rank_dict[node2.item()] = sim_score.item()
    # top_k = sorted(rank_dict.items())[:k]
    top_k = sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)[:k]
    avg_sim = np.mean([value for _, value in top_k])
    print(f'len(top_k): {len(top_k)}')
    print(f'avg_sim: {avg_sim}')
    print(top_k)