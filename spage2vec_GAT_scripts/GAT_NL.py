import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import gc
import time
import pandas as pd
import torch.nn as nn
#from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GATConv
from torch_geometric.nn import Linear
from torch_geometric.utils import degree
from torch_cluster import random_walk
from sklearn.preprocessing import OneHotEncoder

# read mRNA spatial data
data_dir = os.path.join(os.getcwd(), 'data')
df_taglist = pd.read_csv(os.path.join(data_dir, 'taglist_heart.csv'), names=['tag', 'gene'])
df_taglist.loc[len(df_taglist.index)] = ["synthetic","Syn Gene A"]
print(df_taglist)
enc = OneHotEncoder(sparse=False).fit(df_taglist['gene'].to_numpy().reshape(-1, 1))

result_dir = os.path.join(os.getcwd(), 'results')
df_nodes = pd.read_csv(os.path.join(result_dir, 'knn_nodes_psuedo2.csv'), index_col=0)
df_nodes = pd.DataFrame(data=enc.transform(df_nodes['gene'].to_numpy().reshape(-1, 1)), index=df_nodes.index)
df_edges = pd.read_csv(os.path.join(result_dir, 'knn_edges_pseudo2.csv'))

# load into PyG Data
index_dict = dict(zip(df_nodes.index, range(len(df_nodes))))
print(list(df_edges.columns))
#print(index_dict)
df_edges_index = df_edges[['source', 'target']].applymap(index_dict.get)
x = torch.tensor(df_nodes.to_numpy(), dtype=torch.float)
edge_index = torch.tensor(df_edges_index.to_numpy(), dtype=torch.long)
edge_index = to_undirected(edge_index.t().contiguous())
data = Data(x=x, edge_index=edge_index)
print(data)
print(data.num_edges / data.num_nodes)

# hyperparameters
num_samples = [-1, -1] # number of samples in each layer
batch_size = 256
hidden_channels = 32
walk_length = 1
num_neg_samples = 1
epochs = 15
disable = True

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None, 
    sizes=[-1], batch_size=512, shuffle=False, 
    num_workers=4)

class GAT(nn.Module):
        def __init__(self, feat_dim, hidden_dim, pool='mean',  n_layers=2, act='LeakyReLU', heads=1,  bn=True, xavier=True):
            super(GAT, self).__init__()

            if bn:
                self.bns = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            self.acts = torch.nn.ModuleList()
            self.n_layers = n_layers
            self.pool = pool
            if act == 'ELU':
                a = torch.nn.ELU()
            elif act == 'LeakyReLU':
                a = nn.LeakyReLU()
            for i in range(n_layers):
                start_dim = hidden_dim if i else feat_dim
                conv = GATConv(start_dim, hidden_dim, heads=heads, concat=False)
                if xavier:
                    self.weights_init(conv)
                self.convs.append(conv)
                self.acts.append(a)
                if bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))

        def weights_init(self, module):
            for m in module.modules():
                if isinstance(m, GATConv):
                    layers = [m.lin_src, m.lin_dst]
                if isinstance(m, Linear):
                    layers = [m]
                for layer in layers:
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)
    
        def forward(self, x, adj):
            #print(data)
            for i, (edge_index, _, size) in enumerate(adj):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.n_layers - 1:
                    x = F.relu(x)

            return x



        def inference(self, x_all):
            pbar = tqdm(total=x_all.size(0) * self.n_layers)
            for i in range(self.n_layers):
                xs = []
                for batch_size, n_id, adj in subgraph_loader:
                    edge_index, _, size = adj.to(device)
                    x = x_all[n_id].to(device)
                    x_target = x[:size[1]]
                    x = self.convs[i]((x, x_target), edge_index) 
                    #+ self.skips[i](x)
                    if i != self.n_layers - 1:
                        x = F.elu(x)
                    xs.append(x.cpu())

                    pbar.update(batch_size)

                x_all = torch.cat(xs, dim=0)
            pbar.close()
            return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(data.num_node_features, hidden_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = data.x.to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=data.num_nodes, disable=disable)
    pbar.set_description(f'Epoch {epoch:02d}')

    t_loss = 0

    node_idx = torch.randperm(data.num_nodes)
    train_loader = NeighborSampler(
        data.edge_index, node_idx=node_idx, 
        sizes=num_samples, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    
    # positive sampling
    rw = random_walk(data.edge_index[0], data.edge_index[1], node_idx, walk_length=walk_length)
    rw_idx = rw[:,1:].flatten()
    pos_loader = NeighborSampler(
        data.edge_index, node_idx=rw_idx, 
        sizes=num_samples, batch_size=batch_size * walk_length, shuffle=False,
        num_workers=0
    )
    
    # negative sampling
    deg = degree(data.edge_index[0])
    distribution = deg ** 0.75
    neg_idx = torch.multinomial(
        distribution, data.num_nodes * num_neg_samples, replacement=True)
    neg_loader = NeighborSampler(
        data.edge_index, node_idx=neg_idx,
        sizes=num_samples, batch_size=batch_size * num_neg_samples,
        shuffle=True, num_workers=0)

    for (batch_size_, u_id, adjs_u), (_, v_id, adjs_v), (_, vn_id, adjs_vn) in zip(train_loader, pos_loader, neg_loader):
        
        adjs_u = [adj.to(device) for adj in adjs_u]
        z_u = model(x[u_id], adjs_u)

        adjs_v = [adj.to(device) for adj in adjs_v]
        z_v = model(x[v_id], adjs_v)

        adjs_vn = [adj.to(device) for adj in adjs_vn]
        z_vn = model(x[vn_id], adjs_vn)

        optimizer.zero_grad()
        pos_loss = -F.logsigmoid(
            (z_u.repeat_interleave(walk_length, dim=0)*z_v) \
            .sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(
            -(z_u.repeat_interleave(num_neg_samples, dim=0)*z_vn) \
            .sum(dim=1)).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        t_loss += float(loss)
        pbar.update(batch_size_)

    pbar.close()

    loss = t_loss / len(train_loader)

    return loss


loss_lst = []
epoch_lst = []
for epoch in range(1, epochs + 1):
    gc.collect()
    loss = train(epoch)
    if epoch%10==0 or epoch<10:
        loss_lst.append(loss)
        epoch_lst.append(epoch)

    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

model_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(model_dir, exist_ok=True)
model_name = os.path.join(model_dir, time.strftime('gat-nl2-rw-%Y%m%d_pseudo_GAT.pt'))
torch.save(model, model_name)
model = torch.load(model_name)
loss_df = pd.DataFrame(list(zip(epoch_lst, loss_lst)),columns = ['epoch', 'loss'])
gc.collect()
model.eval()
z = model.inference(x)
node_embeddings = z.detach().cpu().numpy()
result_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(result_dir, exist_ok=True)
embedding_name = time.strftime('{}-embedding-%Y%m%dpseudo_GAT.npy'.format(type(model).__name__))
np.save(os.path.join(result_dir, embedding_name), node_embeddings)
loss_df.to_csv(os.path.join(result_dir,'loss_epoch_psuedo2_GAT.csv'))
print(embedding_name)
