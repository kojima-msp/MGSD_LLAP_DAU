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

from util import make_RBF_graph
from methods.MGSD_LLap_DAU import MGSD_LLap_DAU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

if __name__ == '__main__':

    # settings
    N_layers = 9

    datatype = 'Synthetic'
    model_name = 'MGSD_LLap_DAU'

    N_epochs = 30
    N_split = 2

    N_s = 80
    N_m = 120
    noise_list = [0.10, 0.15, 0.2, 0.25, 0.30]
    suffix = f's80m120'


    matfile_list = sorted(glob.glob(f'./data/{datatype}/data_{suffix}/*.mat'))
    N_testfiles = int(len(matfile_list)/N_split)


    trial_list = sorted(glob.glob(f'./data/{datatype}/trained_params/{model_name}/*.pth'))

    rmse_oracle = np.zeros_like(noise_list) # use ground truth graphs
    rmse_proposed = np.zeros_like(noise_list) # use learned graphs
    rmse_learned = np.zeros_like(noise_list) # use predefined graphs
    rmse_ls = np.zeros_like(noise_list) # use ground truth spatial graph
    rmse_lm = np.zeros_like(noise_list) # use ground truth modality graph
    
    ## cross varidation
    for trial_idx, trial in enumerate(trial_list):
        test_matfile_list = matfile_list[trial_idx*N_testfiles:(trial_idx+1)*N_testfiles]

        model = MGSD_LLap_DAU(layers=N_layers, N_s=N_s, N_m=N_m)
        model.load_state_dict(torch.load(f'./data/{datatype}/trained_params/{model_name}/fortest{trial_idx}.pth', weights_only=True))

        model_rbf = MGSD_LLap_DAU(layers=N_layers, N_s=N_s, N_m=N_m)
        model_rbf.load_state_dict(torch.load(f'./data/{datatype}/trained_params/{model_name}_rbf/fortest{trial_idx}.pth', weights_only=True))

        model_gt = MGSD_LLap_DAU(layers=N_layers, N_s=N_s, N_m=N_m)
        model_gt.load_state_dict(torch.load(f'./data/{datatype}/trained_params/{model_name}_gt/fortest{trial_idx}.pth', weights_only=True))

        model_ls = MGSD_LLap_DAU(layers=N_layers, N_s=N_s, N_m=N_m)
        model_ls.load_state_dict(torch.load(f'./data/{datatype}/trained_params/{model_name}_ls/fortest{trial_idx}.pth', weights_only=True))
        
        model_lm = MGSD_LLap_DAU(layers=N_layers, N_s=N_s, N_m=N_m)
        model_lm.load_state_dict(torch.load(f'./data/{datatype}/trained_params/{model_name}_lm/fortest{trial_idx}.pth', weights_only=True))

        for test_idx, test_matfile in enumerate(test_matfile_list):
            TestData = io.loadmat(test_matfile)
            
            X = TestData["X"] # ground truth
            Y_list = torch.tensor(TestData["Y_list"].transpose((2,0,1))) # observed


            for noise_idx, noise in enumerate(noise_list):

                Y = Y_list[noise_idx, : , :]

                # proposed
                _, _, _, X_out = model.forward(Y)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_proposed[noise_idx] += rmse
                print(f'Noise {noise:.2f}\t Proposed RMSE: {rmse:.4f}', end='\t')

                L_s_gt = torch.tensor(TestData["L_s"])
                L_m_gt = torch.tensor(TestData["L_m"])
                L_s_pre, L_m_pre = make_RBF_graph(Y)

                # oracle
                _, _, _, X_out = model_gt.forward(Y, L_m=L_m_gt, L_s=L_s_gt)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_oracle[noise_idx] += rmse
                print(f'Oracle RMSE: {rmse:.4f}', end='\t')

                # predefined
                _, _, _, X_out = model_rbf.forward(Y, L_m=L_m_pre, L_s=L_s_pre)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_learned[noise_idx] += rmse
                print(f'Predefined RMSE: {rmse:.4f}', end='\t')

                # pretrained spatial graph
                _, _, _, X_out = model_ls.forward(Y, L_s=L_s_pre)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_ls[noise_idx] += rmse
                print(f'Pretrained Spatial RMSE: {rmse:.4f}', end='\t')

                # pretrained modality graph
                _, _, _, X_out = model_lm.forward(Y, L_m=L_m_pre)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_lm[noise_idx] += rmse
                print(f'Pretrained Modality RMSE: {rmse:.4f}')
        
    print(rmse_proposed/len(test_matfile_list)*N_split)
    print(rmse_oracle/len(test_matfile_list)*N_split)
    print(rmse_learned/len(test_matfile_list)*N_split)
    print(rmse_ls/len(test_matfile_list)*N_split)
    print(rmse_lm/len(test_matfile_list)*N_split)