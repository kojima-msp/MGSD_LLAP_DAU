"""Train the models using datasets.

Example:

    $ python3 src/train.py Weather ae
    $ python3 src/train.py Weather gcn
    $ python3 src/train.py Weather TGSR_DAU
    $ python3 src/train.py Weather MGSD_LLap_DAU

"""
"""
@Author: Hayate Kojima
@Contact: h-kojima@msp-lab.org
@Date: 2025/04/01
"""

import numpy as np

import torch
import sys

from scipy import io
from torch import optim
import os
import glob

from util import model_selector

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

N_layers = 9

args = sys.argv
datatype = args[1]
model_name = args[2]

N_epochs = 30
N_split = 2

lr = 1e-2

if datatype != 'Synthetic' and datatype != 'Weather':
    sys.exit('Error: datatype should be `Synthetic` or `Weather`')

if datatype == 'Synthetic':
    N_s = 80
    N_m = 120
    noise_list = [0.10, 0.15, 0.2, 0.25, 0.30]
    suffix = f's80m120'

if datatype == 'Weather':
    N_s = 62
    N_m = 73
    noise_list = [3,5,7,9]
    suffix = f'2013-2022'

os.makedirs(f'./data/{datatype}/trained_params/{model_name}', exist_ok=True)

## sort
matfile_list = sorted(glob.glob(f'./data/{datatype}/data_{suffix}/*.mat'))

if len(matfile_list) % N_split != 0:
    sys.exit('Error: Cannot split files. Change the value of N_split')
N_testfiles = int(len(matfile_list)/N_split)

## Cross Validation
for cv_idx in range(N_split):
    test_matfile_list = matfile_list[cv_idx*N_testfiles:(cv_idx+1)*N_testfiles]
    train_matfile_list = sorted(list(set(matfile_list) - set(test_matfile_list)))

    ## model_init
    model = model_selector(model_name, N_layers, N_s, N_m).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.param_groups[0]['capturable'] = True

    for epoch in range(N_epochs):
        for train_matfile_idx, train_matfile in enumerate(train_matfile_list):
            TrainData = io.loadmat(train_matfile)
            X = torch.tensor(TrainData['X'])
            Y_list = torch.tensor(TrainData['Y_list'].transpose((2,0,1))) if datatype == 'Synthetic' else torch.tensor(TrainData['Y_list'])
            
            for noise_idx, noise in enumerate(noise_list):
                Y = Y_list[noise_idx, :, :]

                if model_name != 'MGSD_LLap_DAU':
                    X_out = model.forward(Y)
                else:
                    L_m_out, L_s_out, _, X_out = model.forward(Y)
                    
                loss = criterion(X_out, X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(model_name, f'nepochs{epoch:02d}', train_matfile[-11:-4], f'{noise:.03f}', np.sqrt(loss.cpu().detach().numpy()))
            
    torch.save(model.state_dict(), f'./data/{datatype}/trained_params/{model_name}/fortest{cv_idx}.pth')