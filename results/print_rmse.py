"""各手法のRMSEおよびPSNRの可視化

Example:

    $ python3 plot_rmse_psnr.py Synthetic
    $ python3 plot_rmse_psnr.py Weather

    The table of RMSE is printed in the terminal.
"""

"""
@Author: Hayate Kojima
@Contact: h-kojima@msp-lab.org
@Date: 2025/04/01
"""

import numpy as np
import os
import sys
from scipy import io
from sklearn.metrics import root_mean_squared_error
import pandas as pd

# 共通の設定
method_list = ['glpf', 'hd', 'svds', 'ae', 'gcn', 'TGSR_DAU', 'MGSD_LLap_DAU']
model_type_1 = ['glpf', 'hd', 'svds'] ## fixed layers/iters
N_epochs = 30
N_layers_list = [1,5,9]

args = sys.argv
dataset = args[1]

# データセットごとに異なる設定
if dataset == 'Synthetic':
    noise_list = [0.10, 0.15, 0.20, 0.25, 0.30]
    suffix = 's80m120'
    vmin = 0
    vmax = 1

elif dataset == 'Weather':
    noise_list = [3,5,7,9]
    suffix = '2013-2022'
    vmin = -10
    vmax = 40

path = f'./results/_figures/{dataset}/rmse_psnr/'

df = pd.DataFrame(index=[model_idx.upper().replace('_', '-') for model_idx in method_list], columns=noise_list, dtype=float)
df = df.astype(float).fillna(0.0)

# plot results
for model_name in method_list:

    rmse = np.zeros_like(noise_list)

    N_layers = N_layers_list[0] if model_name in model_type_1 else N_layers_list[-1]

    for data_idx in range(10):

        trial_idx = 0 if data_idx <= 4 else 1

        # original
        data = io.loadmat(f'./data/{dataset}/data_{suffix}/data_{data_idx:02d}.mat')
        X = data['X']

        npz = np.load(f'./results/{dataset}/data_{suffix}/{model_name}/test{trial_idx:02d}_nepochs{N_epochs}.npz')

        for noise_idx, noise in enumerate(noise_list):

            target_X = npz['X_out_list'][data_idx%5, noise_idx] if npz['X_out_list'].ndim==4 else npz['X_out_list'][data_idx%5, noise_idx, -1]
            
            df.loc[model_name.upper().replace('_', '-'), noise_list[noise_idx]] += root_mean_squared_error(X, target_X)

df/=10.0

print(df.to_string(float_format="%.3f"))
# print(df.to_latex(float_format="%.3f", column_format='l' + 'c' * len(df.columns), label=f'table:rmse_{dataset}', caption=''))
