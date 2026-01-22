"""各手法の結果の可視化

Example:

    $ python3 v2_plot_signal.py Synthetic 0
    $ python3 v2_plot_signal.py Weather 0

    where, 0 is the index of the dataset.
    The out put files are saved in the './results/_figures/{dataset}/signal/' directory.
"""

"""
@Author: Hayate Kojima
@Contact: h-kojima@msp-lab.org
@Date: 2025/04/01
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import io
from sklearn.metrics import root_mean_squared_error

# 共通の設定
method_list = ['glpf', 'hd', 'svds', 'ae', 'gcn', 'TGSR_DAU', 'MGSD_LLap_DAU']
model_type_1 = ['glpf', 'hd', 'svds', 'gcn'] ## fixed layers/iters
N_epochs = 30
N_layers_list = [1,5,9]

args = sys.argv
dataset = args[1]
data_idx = int(args[2])

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

os.makedirs(f'./results/_figures/{dataset}/signal/', exist_ok=True)

# original
data = io.loadmat(f'./data/{dataset}/data_{suffix}/data_{data_idx:02d}.mat')
X = data['X']

Y_list = data['Y_list'] if dataset == 'Weather' else data['Y_list'].transpose((2,0,1))

path = f'./results/_figures/{dataset}/signal/'

# plot groundtruth
plt.figure()
sns.heatmap(X, vmin=vmin, vmax=vmax, cmap='jet')
plt.savefig(path+f'{data_idx:02d}_original.pdf', bbox_inches='tight')
plt.close()

f = open(path + f'{data_idx:02d}_rmses.txt', 'w')

# plot noisy data
for noise_idx, noise in enumerate(noise_list):
    plt.figure()
    sns.heatmap(Y_list[noise_idx], vmin=vmin, vmax=vmax, cmap='jet')
    plt.savefig(path+'{data_idx:02d}_obs_noise{noise}.pdf'.format(noise='{:.3f}'.format(noise).replace('.',''), data_idx=data_idx), bbox_inches='tight')
    plt.close()
    print(f'{data_idx:02d}_obs_noise{noise}: {root_mean_squared_error(X, Y_list[noise_idx])}', file=f)


trial_idx = 0 if data_idx <= 4 else 1

# plot results
for model_name in method_list:

    # N_layers_iter = [N_layers_list[0]] if model_name in model_type_1 else N_layers_list
    N_layers = N_layers_list[0] if model_name in model_type_1 else N_layers_list[-1]

    npz = np.load(f'./results/{dataset}/data_{suffix}/{model_name}/test{trial_idx:02d}_nepochs{N_epochs}.npz')

    for noise_idx, noise in enumerate(noise_list):

        target_X = npz['X_out_list'][data_idx%5, noise_idx] if npz['X_out_list'].ndim==4 else npz['X_out_list'][data_idx%5, noise_idx, -1]
        
        plt.figure()
        sns.heatmap(target_X, vmin=vmin, vmax=vmax, cmap='jet')
        plt.savefig(path+'{data_idx:02d}_{model_name}_noise{noise}.pdf'.format(model_name=model_name, noise='{:.3f}'.format(noise).replace('.',''), data_idx=data_idx), bbox_inches='tight')
        plt.close()
        print(f'{data_idx:02d}_{model_name}_noise{noise}: {root_mean_squared_error(X, target_X)}', file=f)

f.close()