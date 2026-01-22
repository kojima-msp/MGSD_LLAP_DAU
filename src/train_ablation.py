import os
import glob
import tqdm

from scipy import io
import torch
from torch import optim

from util import make_RBF_graph
from methods.MGSD_LLap_DAU import MGSD_LLap_DAU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

# 共通の設定
mode = 'lm' 
# 'gt' (for ablation study that uses ground truth graphs) or 'rbf' (for ablation study that uses RBF graphs)
# or 'ls' (for ablation study that uses only spatial learned graph) or  'lm' (for ablation study that uses only temporal learned graph)

datatype = 'Synthetic'
N_s = 80
N_m = 120
noise_list = [0.10, 0.15, 0.2, 0.25, 0.30]
suffix = f's80m120'

# train settings
N_layers = 9
N_epochs = 30
N_split = 2
lr = 1e-2

path = f'./data/{datatype}/trained_params/MGSD_LLap_DAU_{mode}'

os.makedirs(path, exist_ok=True)

## sort
matfile_list = sorted(glob.glob(f'./data/{datatype}/data_{suffix}/*.mat'))

N_testfiles = int(len(matfile_list)/N_split)

## Cross Validation
for cv_idx in range(N_split):
    test_matfile_list = matfile_list[cv_idx*N_testfiles:(cv_idx+1)*N_testfiles]
    train_matfile_list = sorted(list(set(matfile_list) - set(test_matfile_list)))

    ## model_init
    model = MGSD_LLap_DAU(layers=N_layers, N_s=N_s, N_m=N_m)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.param_groups[0]['capturable'] = True

    for epoch in tqdm.tqdm(range(N_epochs)):
        for train_matfile_idx, train_matfile in enumerate(train_matfile_list):
            TrainData = io.loadmat(train_matfile)
            X = torch.tensor(TrainData['X'])
            Y_list = torch.tensor(TrainData['Y_list'].transpose((2,0,1))) if datatype == 'Synthetic' else torch.tensor(TrainData['Y_list'])
            
            for noise_idx, noise in enumerate(noise_list):
                Y = Y_list[noise_idx, :, :]

                L_s, L_m = None, None

                if mode == 'gt':
                    L_s = torch.tensor(TrainData['L_s'])
                    L_m = torch.tensor(TrainData['L_m'])

                else:
                    L_s, L_m = make_RBF_graph(X)

                    if mode == 'ls':
                        L_m = None

                    if mode == 'lm':
                        L_s = None

                    # rbf case: L_s, L_m are already set          
                
                _, _, _, X_out = model.forward(Y, L_m=L_m, L_s=L_s)
                    
                loss = criterion(X_out, X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    torch.save(model.state_dict(), f'{path}/fortest{cv_idx}.pth')