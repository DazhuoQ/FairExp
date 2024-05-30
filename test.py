import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv
import torch.optim as optim

# Load the Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        hidden_channels = 64
        heads = 8

        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6)
        self.conv3 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        test_acc = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)

# Save all node embeddings
torch.save(out, 'node_embeddings.pt')
print("All node embeddings have been saved.")

# Load the embeddings
loaded_embeddings = torch.load('node_embeddings.pt')
print("Loaded embeddings:", loaded_embeddings)

# Retrieve embeddings for nodes 3 and 9 from the loaded embeddings
node_3_embedding = loaded_embeddings[3]
node_9_embedding = loaded_embeddings[9]

print("Embedding for node 3:", node_3_embedding)
print("Embedding for node 9:", node_9_embedding)

# Calculate cosine similarity between embeddings of node 3 and node 9
cosine_sim = F.cosine_similarity(node_3_embedding.unsqueeze(0), node_9_embedding.unsqueeze(0))

print("Cosine similarity between embeddings of node 3 and node 9:", cosine_sim.item())
