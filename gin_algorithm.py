import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GINConv
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d
import torch.optim as optim
from utils import *
import random

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GIN, self).__init__()
        hidden_channels = 64

        self.conv1 = GINConv(Seq(Linear(num_node_features, hidden_channels),
                                 ReLU(),
                                 Linear(hidden_channels, hidden_channels),
                                 ReLU(),
                                 BatchNorm1d(hidden_channels)), train_eps=True)
        
        self.conv2 = GINConv(Seq(Linear(hidden_channels, hidden_channels),
                                 ReLU(),
                                 Linear(hidden_channels, hidden_channels),
                                 ReLU(),
                                 BatchNorm1d(hidden_channels)), train_eps=True)
        
        self.conv3 = GINConv(Seq(Linear(hidden_channels, hidden_channels),
                                 ReLU(),
                                 Linear(hidden_channels, hidden_channels),
                                 ReLU(),
                                 BatchNorm1d(hidden_channels)), train_eps=True)
        
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.lin(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_gin_model(data, seed):

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GIN(data.x.size(1), data.y.size(0)).to(device)
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
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

    # Save the embeddings for later use
    torch.save(out, 'models/gin_test_node_embeddings.pt')


def gin_emb():

    embeddings = torch.load('models/gin_test_node_embeddings.pt')

    return embeddings
