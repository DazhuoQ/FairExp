import torch
import numpy as np
from utils import *
from sklearn.decomposition import PCA
import umap


# Function to compute the hash code
def hash_code(vector, hyperplanes):
    if np.isnan(np.sign(vector @ hyperplanes.T).astype(int).any()):
        print('nmd wr')
    return np.sign(vector @ hyperplanes.T).astype(int)

# Function to get node degree
def get_degree(node_idx, data):
    return data.edge_index[1][data.edge_index[0] == node_idx].size(0)

# Function to compute combined hash code with weighted features and degree
def combined_hash_code(node_idx, hyperplanes, W, data):
    feature_vector = data.x[node_idx].numpy()
    degree = get_degree(node_idx, data)
    combined_vector = W * feature_vector + (1 - W) * degree
    return hash_code(combined_vector, hyperplanes)

# Hierarchical LSH: compute hash codes for multiple hops
def hierarchical_hash_code(node_idx, hyperplanes, W, hops, data):
    current_neighbors = set([node_idx])
    hash_list = []
    
    for decay in range(hops):
        next_neighbors = set()
        for n in current_neighbors:
            next_neighbors.update(data.edge_index[1, data.edge_index[0] == n].tolist())
        # Update current neighbors
        current_neighbors = next_neighbors
        num_samples = int(len(current_neighbors) * np.exp(-decay * hops))
        if num_samples < 1:
            num_samples = 1
        sampled_neighbors = np.random.choice(list(current_neighbors), num_samples, replace=False)
        if sampled_neighbors.all():
            layer_hash = np.mean([combined_hash_code(n, hyperplanes, W, data) for n in sampled_neighbors], axis=0)
            hash_list.append(layer_hash)
    if len(hash_list) == 0:
        hash_code = combined_hash_code(node_idx, hyperplanes, W, data)
    else:
        alpha = 0
        hash_code = alpha * np.mean(hash_list, axis=0) + (1 - alpha) * combined_hash_code(node_idx, hyperplanes, W, data)
    return hash_code


def LSH_index(remaining_nodes, hops, graph, node1, setting, seed, k):

    num_components = 5  # Number of components for PCA

    # Normalize feature vectors
    graph.x = torch.nn.functional.normalize(graph.x, p=2, dim=1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=num_components)
    x_reduced = pca.fit_transform(graph.x.numpy())
    graph.x = torch.tensor(x_reduced, dtype=torch.float)

    # Parameters
    num_hash_functions = 2
    num_candidates = k + 100  # Number of candidates to consider from the hash buckets
    W = 1  # Weight for feature vectors vs. degree
    if setting == 'f':
        W = 1
    np.random.seed(seed) 
    random_hyperplanes = np.random.randn(num_hash_functions, graph.x.size(1))
    hash_codes = np.array([hierarchical_hash_code(i, random_hyperplanes, W, hops, graph) for i in remaining_nodes])
    # Node to find similar nodes for
    query_hash_code = hierarchical_hash_code(node1, random_hyperplanes, W, hops, graph)
    # Filter candidates: find nodes with similar hash codes
    hamming_distances = np.sum(hash_codes != query_hash_code, axis=1)
    candidates_idx = remaining_nodes[np.argsort(hamming_distances)[:num_candidates + 1]] # +1 to include the query node itself
    candidates_idx = candidates_idx[candidates_idx != node1]  # Remove the query node from candidates
    return candidates_idx
