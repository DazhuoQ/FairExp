from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # Get the first graph object

def extract_3_hop_subgraphs(data, num_hops=3):
    subgraph_data_list = []
    test_nodes = data.test_mask.nonzero(as_tuple=True)[0]
    
    for node in test_nodes:
        # Extract the 3-hop subgraph surrounding the test node
        subgraph_node_indices, subgraph_edge_indices, mapping, edge_mask = k_hop_subgraph(
            node_idx=node.item(), num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=True)
        
        # Prepare the subgraph data
        subgraph_features = data.x[subgraph_node_indices]
        subgraph_labels = data.y[subgraph_node_indices]
        subgraph_edge_index = data.edge_index[:, edge_mask]

        # If your graph includes edge features, extract those for the subgraph as well
        subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None

        # Create a new Data object for the subgraph
        subgraph = Data(x=subgraph_features, edge_index=subgraph_edge_index, y=subgraph_labels, edge_attr=subgraph_edge_attr)
        
        # You can also include other data attributes as needed
        # For example, if your graph has a 'train_mask', 'val_mask', or 'test_mask', you can compute them here
        
        subgraph_data_list.append((node, subgraph))
    
    return subgraph_data_list

def generate_unique_pairs(arr):
    unique_pairs = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            unique_pairs.add((arr[i], arr[j]))
    return unique_pairs


# Extract 3-hop subgraphs for test nodes
subgraph_data_list = extract_3_hop_subgraphs(data)

# Example usage
pairs = generate_unique_pairs(subgraph_data_list)

for node1, node2 in range(pairs):
    v1, subg1 = node1
    v2, subg2 = node2
    