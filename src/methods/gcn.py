import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch import nn
from torch.nn.modules.module import Module

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 512)
        self.gc2 = GraphConvolution(512, nclass)

    def forward(self, x):
        adj = self.preprocess(x)
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)

        return x
    
    def preprocess(self, Y):

        N_s,_ = Y.shape

        # RBF
        Z_s = torch.cdist(Y, Y) ** 2
        Z_s = (Z_s - torch.min(Z_s)) / (torch.max(Z_s) - torch.min(Z_s))
        Z_s = Z_s * 10
        W_s = torch.mul(torch.exp(-Z_s / torch.tensor(1)), torch.ones((N_s,N_s))-torch.eye(N_s))
        D_s = torch.squeeze(W_s @ torch.ones(N_s,1))
        W_s_tilde = W_s + torch.eye(N_s)
        D_s_tilde = D_s + torch.ones(N_s)
        P_s_tilde = torch.mm(torch.diag(torch.pow(D_s_tilde, -0.5)), torch.mm(W_s_tilde, torch.diag(torch.pow(D_s_tilde, -0.5))))

        return P_s_tilde
