"""提案手法で学習されたパラメータの可視化

Example:

    $ python3 v2_plot_parameters.py synthetic
    $ python3 v2_plot_parameters.py realdata

    The out put files are saved in the './results/_figures/{dataset}/parameters/' directory.

"""
"""
@Author: Hayate Kojima
@Contact: h-kojima@msp-lab.org
@Date: 2025/04/01
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import torch

plt.style.use(['science','ieee','grid','high-vis'])

if __name__ == '__main__':

    target_idx = 1
    N_layers = 9
    mk_size = 4
    facecolor = "none"
    args = sys.argv
    dataset = args[1]
    path = f'./results/_figures/{dataset}/'


    os.makedirs(path, exist_ok=True)

    model = torch.load(f'./data/{dataset}/trained_params/MGSD_LLap_DAU/fortest{target_idx}.pth', weights_only=True)

    T = np.arange(N_layers)

    alpha_s = np.array([model[f'alpha_s.{t}'].data.cpu().detach().numpy() for t in T]).ravel()
    beta_s = np.array([model[f'beta_s.{t}'].data.cpu().detach().numpy() for t in T]).ravel()
    gamma_s = np.array([model[f'gamma_s.{t}'].data.cpu().detach().numpy() for t in T]).ravel()
    alpha_m = np.array([model[f'alpha_m.{t}'].data.cpu().detach().numpy() for t in T]).ravel()
    beta_m = np.array([model[f'beta_m.{t}'].data.cpu().detach().numpy() for t in T]).ravel()
    gamma_m = np.array([model[f'gamma_m.{t}'].data.cpu().detach().numpy() for t in T]).ravel()

    fig, ax = plt.subplots()
    for params, label, marker, color in zip(
        [alpha_s, beta_s, gamma_s, alpha_m, beta_m, gamma_m],
        [r'$\alpha_s$', r'$\beta_s$', r'$\gamma_s$', r'$\alpha_m$', r'$\beta_m$', r'$\gamma_m$'],
        ['o', '<', '*', '>', 's', 'd'],
        plt.rcParams['axes.prop_cycle'].by_key()['color']
    ):
        ax.plot(T, params, label=label, marker=marker, markersize=mk_size, fillstyle=facecolor, linestyle='solid', color=color)

        # plot linear fit line
        z = np.poly1d(np.polyfit(T, params, 1))
        ax.plot(T, z(T), linestyle='--', color=color, alpha=0.5)



    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    ax.set_xlabel('Layer')

    fig.savefig(path + f'{dataset}_MGSD_LLap_DAU_params.pdf')