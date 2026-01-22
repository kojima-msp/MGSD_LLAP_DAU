import torch

from methods.TGSR_DAU import TGSR_DAU
from methods.gcn import GCN
from methods.ae import AutoEncoder
from methods.MGSD_LLap_DAU import MGSD_LLap_DAU

def model_selector(model_name, N_layers, N_s, N_m):
    if model_name == 'TGSR_DAU':
        return TGSR_DAU(layers=N_layers)

    elif model_name == 'gcn':
        return GCN(N_m, N_m)

    elif model_name == 'ae':
        return AutoEncoder(N_s*N_m, N_layers, BIAS=256)
    
    elif model_name == 'MGSD_LLap_DAU':
        return MGSD_LLap_DAU(layers=N_layers, N_s=N_s, N_m=N_m)
    
def make_RBF_graph(Y):

    # for spatial graph
    N_s = Y.shape[0]
    Z_s = torch.cdist(Y, Y) ** 2
    Z_s = (Z_s - torch.min(Z_s)) / (torch.max(Z_s) - torch.min(Z_s))
    W_s = torch.mul(torch.exp(-Z_s / torch.tensor(1)), torch.ones((N_s, N_s))-torch.eye(N_s))
    D_s = torch.squeeze(W_s @ torch.ones(N_s, 1))
    L_s = torch.diag(D_s) - W_s

    # for temporal graph
    Y = Y.T
    N_m = Y.shape[0]
    Z_m = torch.cdist(Y, Y) ** 2
    Z_m = (Z_m - torch.min(Z_m)) / (torch.max(Z_m) - torch.min(Z_m))
    W_m = torch.mul(torch.exp(-Z_m / torch.tensor(1)), torch.ones((N_m, N_m))-torch.eye(N_m))
    D_m = torch.squeeze(W_m @ torch.ones(N_m, 1))
    L_m = torch.diag(D_m) - W_m 

    return L_s, L_m