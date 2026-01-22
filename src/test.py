"""
$python3 src/test.py
"""

import sys
import os
import glob

import numpy as np
from scipy import io
from sklearn.metrics import root_mean_squared_error

import torch

from util import model_selector

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

if __name__ == '__main__':

    # settings
    N_layers = 9

    args = sys.argv
    datatype = args[1]
    model_name = args[2]

    N_epochs = 30
    N_split = 2

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

    matfile_list = sorted(glob.glob(f'./data/{datatype}/data_{suffix}/*.mat'))
    N_testfiles = int(len(matfile_list)/N_split)


    trial_list = sorted(glob.glob(f'./data/{datatype}/trained_params/{model_name}/*.pth'))

    os.makedirs(f'./results/{datatype}/data_{suffix}/{model_name}', exist_ok=True)
    
    ## cross varidation
    for trial_idx, trial in enumerate(trial_list):
        test_matfile_list = matfile_list[trial_idx*N_testfiles:(trial_idx+1)*N_testfiles]

        model = model_selector(model_name, N_layers, N_s, N_m).to(device)
        model.load_state_dict(torch.load(f'./data/{datatype}/trained_params/{model_name}/fortest{trial_idx}.pth', weights_only=True))

        X_out_list = np.zeros((N_testfiles, len(noise_list), N_s, N_m))

        if model_name == 'MGSD_LLap_DAU':
            X_history_list = np.zeros((N_testfiles, len(noise_list), N_layers+1, N_s, N_m))
            L_s_out_list = np.zeros((N_testfiles, len(noise_list), N_layers+1, N_s, N_s))
            L_m_out_list = np.zeros((N_testfiles, len(noise_list), N_layers+1, N_m, N_m))
            X_groundtruth_list = np.zeros((N_testfiles, N_s, N_m))
            
        for test_idx, test_matfile in enumerate(test_matfile_list):
            TestData = io.loadmat(test_matfile)
            X = TestData["X"]
            Y_list = torch.tensor(TestData["Y_list"].transpose((2,0,1))) if datatype == 'Synthetic' else torch.tensor(TestData["Y_list"])

            for noise_idx, noise in enumerate(noise_list):
                Y = Y_list[noise_idx, : , :]
                
                if model_name != 'MGSD_LLap_DAU':
                    X_out = model.forward(Y)
                else: # MGSD_LLap_DAU
                    L_m_out, L_s_out, X_history, X_out = model.forward(Y)

                rmse = root_mean_squared_error(X, X_out.cpu().detach().numpy())
                print(model_name, 'test{:02d}'.format(trial_idx*N_testfiles+test_idx), f'{noise:.3f}', rmse)     

                X_out_list[test_idx, noise_idx, :, :] = X_out.cpu().detach().numpy()

                if model_name == 'MGSD_LLap_DAU':
                    L_m_out_list[test_idx, noise_idx, :, :, :] = L_m_out.cpu().detach().numpy()
                    L_s_out_list[test_idx, noise_idx, :, :, :] = L_s_out.cpu().detach().numpy()
                    X_history_list[test_idx, noise_idx, :, :, :] = X_history.cpu().detach().numpy()
                    X_groundtruth_list[test_idx, :, :] = X

        outpath = f'./results/{datatype}/data_{suffix}/{model_name}/test{trial_idx:02d}_nepochs{N_epochs}'
        if model_name == 'MGSD_LLap_DAU':
            np.savez_compressed(outpath,
                        X_out_list=X_out_list,
                        X_history_list=X_history_list,
                        L_m_out_list=L_m_out_list,
                        L_s_out_list=L_s_out_list,
                        X_groundtruth_list=X_groundtruth_list
                        )
        else:
            np.savez_compressed(outpath, X_out_list=X_out_list)