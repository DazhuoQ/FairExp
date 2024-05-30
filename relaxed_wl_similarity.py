import torch
import torch.nn.functional as F

def gaussian_kernel(x, y, gamma):
    return torch.exp(-gamma * torch.norm(x - y)**2)

def update_node_features(data, gamma):
    x, edge_index = data.wl_x, data.edge_index
    num_nodes = x.size(0)
    new_features = torch.zeros_like(x)

    # Sum weighted features of neighbors
    for i in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == i]  # Nodes where source nodes are i
        for neighbor in neighbors:
            weight = gaussian_kernel(x[i], x[neighbor], gamma)
            new_features[i] += weight * x[neighbor]

    # Normalize features to prevent numerical instability
    new_features = (new_features) / (new_features.norm(dim=1, keepdim=True) + 1e-6)
    data.wl_x = new_features


def aggregate_features(data, iterations, gamma):
    for _ in range(iterations):
        update_node_features(data, gamma)
        # Aggregate features by storing them
        if 'agg_x' in data:
            data.agg_x += data.wl_x
        else:
            data.agg_x = data.wl_x.clone()

def relaxed_sim_wl(data1, data2, n1, n2):
    kernel_value = F.cosine_similarity(data1.agg_x[n1].float(), data2.agg_x[n2].float())
    return kernel_value
