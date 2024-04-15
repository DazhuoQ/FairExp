# import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import to_undirected
# from wl_similarity import sim_wl


# data1 = Data(
#     edge_index = to_undirected(torch.tensor([[0, 0, 0, 1, 2, 2, 2],
#                                [1, 2, 3, 3, 3, 4, 5]])),
#     y = torch.tensor([5, 2, 4, 3, 1, 1])
# )

# data2 = Data(
#     edge_index = to_undirected(torch.tensor([[0, 0, 1, 1, 2, 2, 3],
#                                [1, 2, 2, 3, 3, 4, 5]])),
#     y = torch.tensor([2, 5, 4, 3, 1, 2])
# )

# result = sim_wl(data1, data2, 1)
# print(result)

my_dict = {'banana': 3, 'apple': 2, 'pear': 1, 'orange': 4, 'mango': 5, 'kiwi': 6}

# Sort the dictionary by keys and get the top 5
top_5_items = sorted(my_dict.items())[:5]

# Convert back to dictionary if needed
top_5_dict = dict(top_5_items)

print("Top 5 items:", top_5_items)
print("Top 5 dict:", top_5_dict)
