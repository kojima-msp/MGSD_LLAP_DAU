"""提案手法の各層で推定されたグラフラプラシアン行列の可視化

Example:

    $ python3 v2_plot_laplacian.py Synthetic 0
    $ python3 v2_plot_laplacian.py Weather 0

    where, 0 is the index of the dataset.
    The out put files are saved in the './results/_figures/{dataset}/laplacian/' directory.

"""
"""
@Author: Hayate Kojima
@Contact: h-kojima@msp-lab.org
@Date: 2025/04/01
"""

import os
import sys

import numpy as np
from scipy import io

import matplotlib.pyplot as plt
import seaborn as sns

# 共通の設定
N_layers = 9

args = sys.argv
dataset = args[1]
data_idx = int(args[2])
N_epochs = 30
vmin = 0
vmax = 1

# データセットごとに異なる設定
if dataset == 'Synthetic':
    noise_list = [0.10, 0.15, 0.20, 0.25, 0.30]
    suffix = 's80m120'

elif dataset == 'Weather':
    noise_list = [3,5,7,9]
    suffix = '2013-2022'

path = f'./results/_figures/{dataset}/laplacian/{data_idx:02d}_'
os.makedirs(f'./results/_figures/{dataset}/laplacian/', exist_ok=True)


# plot groundtruth graph
if dataset == 'Synthetic':
    data = io.loadmat(f'./data/{dataset}/data_{suffix}/data_{data_idx:02d}.mat')

    W_s_gt = data['W_s']
    W_m_gt = data['W_m']

    plt.figure()
    sns.heatmap(W_s_gt, vmin=vmin, vmax=vmax, cmap='jet')
    plt.savefig(path+f'Ws_gt.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    sns.heatmap(W_m_gt, vmin=vmin, vmax=vmax, cmap='jet')
    plt.savefig(path+f'Wm_gt.pdf', bbox_inches='tight')
    plt.close()

# plot estimated graph
trial_idx = 0 if data_idx <= 4 else 1

npz = np.load(f'./results/{dataset}/data_{suffix}/MGSD_LLap_DAU/test{trial_idx:02d}_nepochs{N_epochs}.npz')

L_s_out_list = npz['L_s_out_list']
L_m_out_list = npz['L_m_out_list']

for noise_idx, noise in enumerate(noise_list):

    for layer in range(N_layers+1):
        L_s = L_s_out_list[data_idx%5, noise_idx, layer, :, :]
        L_m = L_m_out_list[data_idx%5, noise_idx, layer, :, :]
        D_s = np.multiply(L_s, np.eye(L_s.shape[0]))
        D_m = np.multiply(L_m, np.eye(L_m.shape[0]))
        W_s = D_s - L_s
        W_m = D_m - L_m
    
        plt.figure()
        sns.heatmap(W_s, vmin=vmin, vmax=vmax, cmap='jet')
        plt.savefig(path+f'MGSD_LLap_DAU_Ws_layer{layer}_noise{"{:.3f}".format(noise).replace(".","")}.pdf', bbox_inches='tight')
        plt.close()

        plt.figure()
        sns.heatmap(W_m, vmin=vmin, vmax=vmax, cmap='jet')
        plt.savefig(path+f'MGSD_LLap_DAU_Wm_layer{layer}_noise{"{:.3f}".format(noise).replace(".","")}.pdf', bbox_inches='tight')
        plt.close()