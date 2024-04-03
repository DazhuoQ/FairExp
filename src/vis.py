import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import torch_geometric.utils as gm_utils

import Preprocessing as dpp
import networkx as nx
import scipy.sparse as sp
from torch_geometric.datasets import KarateClub
from tqdm.std import tqdm


# 定义最后可视化的函数
def visualize(h, color, nodelist=None, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    if nodelist:
        color = [color[i] for i in nodelist]

        # 创建一个只包含子集节点的新图
        h = h.subgraph(nodelist)
 
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=True,
                         node_color=color, cmap="Set2")
    plt.show()

if __name__ == "__main__":
    data_path_root = './'
    adj, features, labels, idx_train_list, idx_val_list, idx_test_list, sens, sens_idx, raw_data_info = dpp.load_data(data_path_root, "bail")
    if raw_data_info is None:
        raw_data_info = {'adj': adj}
    # dpp.pre_analysis(adj, labels, sens)
    adj = adj[:200, :200]
    adj = adj - sp.eye(adj.shape[0])
    features = features[:200]
    labels = labels[:200]
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    num_class = labels.unique().shape[0] - 1

    # preprocess the input
    
    n = features.shape[0]
    data = Data(x=features, edge_index=edge_index)
    data.y = labels
    
    base_node = 12
    G = to_networkx(data, to_undirected=True)

    base_graph = nx.ego_graph(G, base_node, radius=2)

    visualize(base_graph, ['tab:blue' if i == 0 else "tab:green" for i in data.y[list(base_graph.nodes())]])
    for i in tqdm(range(n)):
        
        if sens[i] != sens[base_node] and labels[i] != labels[base_node]:
            ego_subgraph = nx.ego_graph(G, i, radius=2)
            path, dis = nx.optimal_edit_paths(base_graph, ego_subgraph)
            print(f"{i}:{dis}")
            if dis <= 3:
                print(i)
                visualize(ego_subgraph, ['tab:blue' if i == 0 else "tab:green" for i in data.y[list(ego_subgraph.nodes())]])