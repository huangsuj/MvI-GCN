"""
@Project   : HGCN-Net
@Time      : 2021/11/7
@Author    : Zhihao Wu
@File      : model.py
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys



class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=4):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 2, bias=False),
                nn.Tanh(),
                nn.Linear(2, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb

def ortho_norm(weight):
    wtw = torch.mm(weight.t(), weight) + 1e-4 * torch.eye(weight.shape[1]).to(weight.device)
    L = torch.linalg.cholesky(wtw)
    weight_ortho = torch.mm(weight, L.inverse().t())
    return weight_ortho

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, device, activation=F.relu, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.device = device
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        if self.activation==None:
            return x
        else:
            return self.activation(x)


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


class MvGCN(nn.Module):
    def __init__(self, n, num_class, num_views, hidden_dims, dropout, adj_list, device):
        super(MvGCN, self).__init__()
        self.device = device
        self.n = n
        self.num_views = num_views
        self.num_class = num_class
        self.device = device
        self.dropout = dropout
        self.adj_list = adj_list
        self.gc = nn.ModuleList()
        self.hidm = hidden_dims

        for i in range(len(hidden_dims) - 2):
            self.gc.append(GraphConv(hidden_dims[i], hidden_dims[i + 1], self.device))
        self.gc.append(GraphConv(hidden_dims[-2], hidden_dims[-1], self.device, None))
        self.fusion_module = FusionLayer(num_views, 'attention', num_class)

    def forward(self, co_feature, adj_list):
        output_list = []
        for i in range(len(adj_list)):
            output = co_feature
            adj = adj_list[i]
            for gc in self.gc:
                output = gc(output, adj)
                tmp = len(self.hidm) - 1
                if self.dropout != 0 and tmp > 1:
                    output = F.dropout(output, self.dropout, training=self.training)
            output_list.append(output)
        sum_output = self.fusion_module(output_list)
        return sum_output

    def thred_proj(self, theta):
        theta_sigmoid = torch.sigmoid(theta)
        theta_sigmoid_mat = theta_sigmoid.repeat(1, theta_sigmoid.shape[0])
        theta_sigmoid_triu = torch.triu(theta_sigmoid_mat)
        theta_sigmoid_diag = torch.diag(theta_sigmoid_triu.diag())
        theta_sigmoid_tri = theta_sigmoid_triu + theta_sigmoid_triu.t() - theta_sigmoid_diag
        return theta_sigmoid_tri
