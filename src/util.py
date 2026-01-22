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

    def _main(Y):
        # for spatial graph
        N_e = Y.shape[0]
        Z_e = torch.cdist(Y, Y) ** 2
        Z_e = (Z_e - torch.min(Z_e)) / (torch.max(Z_e) - torch.min(Z_e))
        W_e = torch.mul(torch.exp(-Z_e / torch.tensor(1)), torch.ones((N_e, N_e))-torch.eye(N_e))
        D_e = torch.squeeze(W_e @ torch.ones(N_e, 1))
        return torch.diag(D_e) - W_e

    return _main(Y), _main(Y.T)