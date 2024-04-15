from tqdm.std import tqdm
import torch
from torch_geometric.utils import k_hop_subgraph
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def neighbor_aggregate(g, node, label, wl_labels, flag):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(node, 1, g.edge_index)
    mask = subset != node
    result_subset = subset[mask]
    if flag != 0:
        result_subset += flag
    agg_label = np.sort(wl_labels[result_subset].numpy())
    neighbor_labels = ''.join(agg_label.astype(str))
    return (str(label.item()), neighbor_labels)


def origin_label_count(label_vector, l_num):
    counts = torch.bincount(label_vector, minlength=l_num+1)
    counts = counts[1:l_num+1]
    return counts


def wl_label_count(wl_labels, g_identifier):
    unique_hashes, inverse_indices = torch.unique(wl_labels, return_inverse=True)
    # Split the inverse_indices tensor
    former = inverse_indices[:g_identifier]
    latter = inverse_indices[g_identifier:]
    # Calculate counts for the former part
    former_counts = torch.bincount(former, minlength=len(unique_hashes))
    # Calculate counts for the latter part
    latter_counts = torch.bincount(latter, minlength=len(unique_hashes))
    return (former_counts, latter_counts)


def sim_wl(g1, g2, n_iter):
    g_identifier = g1.wl_y.size(0)
    global_wl_labels = torch.cat((g1.wl_y, g2.wl_y))
    wl_labels = deepcopy(global_wl_labels)

    # update the wl_labels with new labels
    for i in range(n_iter):
        for idx, label in enumerate(wl_labels):
            if idx < g_identifier:
                flag = 0
                wl_labels[idx] = hash(neighbor_aggregate(g1, idx, label, global_wl_labels, flag))
            else:
                flag = g_identifier
                wl_labels[idx] = hash(neighbor_aggregate(g2, idx-g_identifier, label, global_wl_labels, flag))
        global_wl_labels = wl_labels
    
    former_counts, latter_counts = wl_label_count(global_wl_labels, g_identifier)
    g1_vector = torch.cat((origin_label_count(g1.wl_y, 5), former_counts))
    g2_vector = torch.cat((origin_label_count(g2.wl_y, 5), latter_counts))
    # similarity = torch.dot(g1_vector, g2_vector)
    similarity = F.cosine_similarity(g1_vector.float(), g2_vector.float(), dim=0)

    return similarity