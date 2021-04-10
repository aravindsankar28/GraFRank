import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from models.GraFRank import GraFrank
from utils.sampler import NeighborSampler
from models.SAGE import SAGE

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
model_type = 'GraFrank'  # Use 'SAGE' to run GraphSAGE.


# generate dummy edge features (5-dimensional vector of ones).
n_edge_channels = 5
data.edge_attr = torch.ones([data.edge_index.shape[1], n_edge_channels])  #

train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
                               shuffle=True, num_nodes=data.num_nodes)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_type == 'GraFrank':
    model = GraFrank(data.num_node_features, hidden_channels=64, edge_channels=n_edge_channels, num_layers=2,
                     input_dim_list=[350, 350, 350, 383])  # input dim list assumes that the node features are first
    # partitioned and then concatenated across the K modalities.
else:
    model = SAGE(data.num_node_features, hidden_channels=64, num_layers=2)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
x = data.x.to(device)


def train(loader):
    model.train()

    total_loss = 0
    it = 0
    for batch_size, n_id, adjs in loader:
        it += 1
        edge_attrs = [data.edge_attr[e_id] for (edge_index, e_id, size) in adjs]
        adjs = [adj.to(device) for adj in adjs]
        edge_attrs = [edge_attr.to(device) for edge_attr in edge_attrs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs, edge_attrs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        # binary skipgram loss can be replaced with margin-based pairwise ranking loss.
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
    model.eval()
    out = model.full_forward(x, edge_index, edge_attr).cpu()
    return out


for epoch in range(1, 51):
    loss = train(train_loader)
    test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
