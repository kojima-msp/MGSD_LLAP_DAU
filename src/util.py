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