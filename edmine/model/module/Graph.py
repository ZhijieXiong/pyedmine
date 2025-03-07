import torch
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class RCDGraphLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(RCDGraphLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class HyperCDgraph:
    def __init__(self, H: np.ndarray):
        self.H = H
        # avoid zero
        self.Dv = np.count_nonzero(H, axis=1) + 1
        self.De = np.count_nonzero(H, axis=0) + 1

    def to_tensor_nadj(self):
        coo = sp.coo_matrix(self.H @ np.diag(1 / self.De) @ self.H.T @ np.diag(1 / self.Dv))
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()