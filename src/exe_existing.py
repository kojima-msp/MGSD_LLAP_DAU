import torch
import numpy as np
import pandas as pd
import scipy.io as io
import os
import glob
import sys
from methods.glpf import GLPF
from methods.hd import HD
from methods.svds import SVDS

# 共通の設定
N_epochs = 30
N_layers = 1
N_split = 2
args = sys.argv
datatype = args[1]
model_name = args[2]

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

if model_name == "glpf":
    
    model_class = GLPF
    
    if datatype == 'Synthetic':
        parameters_list = [0.035583, 0.042912]
    
    if datatype == 'Weather':
        parameters_list = [0.087353, 0.056902]

if model_name == "hd":
    model_class = HD
    
    if datatype == 'Synthetic':
        parameters_list = [48.320637, 0.584129]
    
    if datatype == 'Weather':
        parameters_list = [0.034757, 1.497691]

if model_name == "svds":
    model_class = SVDS
    
    if datatype == 'Synthetic':
        parameters_list = [247.605333, 646.589087]
    
    if datatype == 'Weather':
        parameters_list = [6674.063131, 6769.861368]

# load data
matfile_list = sorted(glob.glob(f'./data/{datatype}/data_{suffix}/*.mat'))

if len(matfile_list) % N_split != 0:
    sys.exit('Error: Cannot split files. Change the value of N_split')
N_testfiles = int(len(matfile_list)/N_split)


loss_df = pd.DataFrame(np.zeros((len(noise_list), N_split)),
                        columns = [f'test_{file_idx:02d}' for file_idx in range(N_split)],
                        index = ['{noise:03f}' for noise in noise_list]
                        )
os.makedirs(f'./results/{datatype}/data_{suffix}/nlayers{N_layers:02d}/{model_name}', exist_ok=True)

## cross varidation
for trial_idx in range(N_split):
    
    model = model_class(parameters_list[trial_idx])

    test_matfile_list = matfile_list[trial_idx*N_testfiles:(trial_idx+1)*N_testfiles]

    X_out_list = np.zeros((N_testfiles, len(noise_list), N_s, N_m))

    for test_idx, test_matfile in enumerate(test_matfile_list):
        TestData = io.loadmat(test_matfile)

        X = torch.tensor(TestData["X"])
        Y_list = torch.tensor(TestData["Y_list"].transpose((2,0,1))) if datatype == 'Synthetic' else torch.tensor(TestData["Y_list"])

        for noise_idx, noise in enumerate(noise_list):
            Y = Y_list[noise_idx, :, :]
            X_out = model.forward(Y)
            loss = torch.mean((X-X_out)**2)
            print(model_name, f'nlayers{N_layers:02d}', 'test{:02d}'.format(trial_idx*N_testfiles+test_idx), f'{noise:.3f}', np.sqrt(loss.cpu().detach().numpy()))     
            loss_df.iloc[noise_idx, trial_idx] = loss_df.iloc[noise_idx, trial_idx] + np.sqrt(loss.cpu().detach().numpy()) / N_testfiles
            X_out_list[test_idx, noise_idx, :, :] = X_out.cpu().detach().numpy()

    # save as csv                
    np.savez_compressed(f'./results/{datatype}/data_{suffix}/nlayers{N_layers:02d}/{model_name}/test{trial_idx:02d}_nlayers{N_layers}_nepochs{N_epochs}',
                        X_out_list=X_out_list
                        )
    loss_df.to_csv(f'./results/{datatype}/data_{suffix}/nlayers{N_layers:02d}/{model_name}/nlayers{N_layers}_nepochs{N_epochs}.csv')