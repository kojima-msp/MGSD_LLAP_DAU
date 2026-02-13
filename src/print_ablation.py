"""Print the RMSE results for ablation study.

Example:

    $ python3 src/print_ablation.py

"""
"""
@Author: Hayate Kojima
@Contact: h-kojima@msp-lab.org
@Date: 2025/12/26
"""

import glob
from tqdm import tqdm

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

    rmse_observed = np.zeros_like(noise_list) # observed data
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

        for test_idx, test_matfile in enumerate(tqdm(test_matfile_list)):
            TestData = io.loadmat(test_matfile)
            
            X = TestData["X"] # ground truth
            Y_list = torch.tensor(TestData["Y_list"].transpose((2,0,1))) # observed


            for noise_idx, noise in enumerate(noise_list):

                Y = Y_list[noise_idx, : , :]

                # observed
                rsme = root_mean_squared_error(X, Y.detach().cpu().numpy())
                rmse_observed[noise_idx] += rsme

                # proposed
                _, _, _, X_out = model.forward(Y)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_proposed[noise_idx] += rmse

                L_s_gt = torch.tensor(TestData["L_s"])
                L_m_gt = torch.tensor(TestData["L_m"])
                L_s_pre, L_m_pre = make_RBF_graph(Y)

                # oracle
                _, _, _, X_out = model_gt.forward(Y, L_m=L_m_gt, L_s=L_s_gt)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_oracle[noise_idx] += rmse

                # predefined
                _, _, _, X_out = model_rbf.forward(Y, L_m=L_m_pre, L_s=L_s_pre)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_learned[noise_idx] += rmse

                # pretrained spatial graph
                _, _, _, X_out = model_ls.forward(Y, L_s=L_s_pre)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_ls[noise_idx] += rmse

                # pretrained modality graph
                _, _, _, X_out = model_lm.forward(Y, L_m=L_m_pre)
                rmse = root_mean_squared_error(X, X_out.detach().cpu().numpy())
                rmse_lm[noise_idx] += rmse
    
    print('Observed', rmse_observed/len(test_matfile_list)/N_split)
    print('Oracle', rmse_oracle/len(test_matfile_list)/N_split)
    print('Learned', rmse_learned/len(test_matfile_list)/N_split)
    print('Learned L_s', rmse_ls/len(test_matfile_list)/N_split)
    print('Learned L_m', rmse_lm/len(test_matfile_list)/N_split)
    print('Proposed', rmse_proposed/len(test_matfile_list)/N_split)