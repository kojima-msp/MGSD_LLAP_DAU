"""Plot Layer vs RMSE.

Example:

    $ python3 plot_layer_vs_rmse.py

    The out put files are saved in the './results/_figures/Synthetic/' directory.
"""

"""
@Author: Hayate Kojima
@Contact: h-kojima@msp-lab.org
@Date: 2025/12/23
"""

import numpy as np
from sklearn.metrics import root_mean_squared_error
import pandas as pd

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams["font.size"] = 12

# 共通の設定
N_epochs = 30
N_layers = 9
dataset = 'Synthetic'

# データセットごとに異なる設定
noise_list = [0.10, 0.15, 0.20, 0.25, 0.30]
suffix = 's80m120'

path = f'./results/_figures/{dataset}/'

df = pd.DataFrame(index=np.arange(N_layers), columns=noise_list, dtype=float)
df = df.astype(float).fillna(0.0)

# plot results

for trial_idx in range(2):

    npz = np.load(f'./results/{dataset}/data_{suffix}/MGSD_LLap_DAU/test{trial_idx:02d}_nepochs{N_epochs}.npz')

    target_X = npz['X_history_list']
    gt_X = npz['X_groundtruth_list']
    
    for n in range(target_X.shape[1]):
        for idx, noise in enumerate(noise_list):
            for t in range(N_layers):
                df.loc[t, noise] += root_mean_squared_error(gt_X[n], target_X[n,idx,t+1])
    
df/=10

# plot
plt.figure(figsize=(6,4))

# 各ノイズレベルごとにプロット
for noise in noise_list:
    plt.plot(df.index+1, df[noise], marker='o', label=fr'$\sigma$: {noise:.2f}')

# 平均をプロット
mean_rmse = df.mean(axis=1)
plt.plot(df.index+1, mean_rmse, marker='s', color='black', linestyle='--', label='Mean')

plt.xticks(np.arange(1, N_layers+1, 1))
plt.xlabel('Layer')
plt.ylabel('RMSE')
plt.legend()
plt.tight_layout()
plt.savefig(path+'MGSD_LLap_DAU_layers_vs_rmse.pdf', bbox_inches='tight')
plt.close()