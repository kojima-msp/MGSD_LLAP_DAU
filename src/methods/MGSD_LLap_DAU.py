"""
 Multimodal Graph Signal Denoising with simultaneous Learning Laplacian Matrix using Deep Algorithm Unrolling
 written by takanami
"""

import numpy as np

import torch
import sys

from torch import nn
from torch.nn import functional as F
from scipy import sparse
from scipy import io
from typing import Optional


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)



class MGSD_LLap_DAU(nn.Module):
    def __init__(self, layers:int, N_s:int, N_m:int, default:Optional[float]=0.5, step_size_pds:float=5e-3, iters_pds:int=1000, tol_pds:float=1e-3, coefficient_L=0.1, run_debug:bool=False):
        super().__init__()
        self.layers = layers
        self.alpha_s = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)])
        self.beta_s = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)])
        self.gamma_s = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)])
        self.alpha_m = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)])
        self.beta_m = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)])
        self.gamma_m = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)])
        self.step_size_pds = step_size_pds
        self.iters_pds = iters_pds
        self.tol_pds = tol_pds
        self.coefficient_L = coefficient_L
        self.run_debug = run_debug
        self.N_m = N_m
        self.N_s = N_s
        self.Phi_m = self._create_Phi(N_m)
        self.Phi_s = self._create_Phi(N_s)
        self.Psi_m = self._create_Psi(N_m)
        self.Psi_s = self._create_Psi(N_s)
    
    ## ReLU
    def _relu(self, x):
        return torch.maximum(x, torch.tensor(1e-08))
    
    def _min_max_normalization(self, x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    
    def _calc_rmse(self, Y, X):
        return torch.sqrt(torch.mean((Y-X)**2))

    ## convert matrix to half-vector
    def _mat2vech(self, L):
        N = L.shape[0]
        ell = torch.zeros(int(N*(N-1)/2), 1)
        k = 0
        for col in range(N):
            for row in range(col+1, N):
                ell[k] = L[row, col]
                k = k + 1
        return ell
    
    ## arange transformation matrix that convert vec(L) from vech(L)
    def _create_Phi(self, N):
        Phi = torch.zeros(N**2, int(N*(N-1)/2))
        k = 0   ## k=N*col+row と表せるが，理解の簡単のためkを使う．row<colのときにこの考えを利用する．
        for col in range(N):
            for row in range(N):
                if row > col:
                    Phi[k, k-int((col+1)*(col+2)/2)] = 1
                    Phi[int(k/N)*(N+1), k-int((col+1)*(col+2)/2)] = -1  ## あとで変更
                elif row < col:
                    Phi[k, col-1+int((N-2+N-row-1)*row/2)] = 1
                    Phi[int(k/N)*(N+1), col-1+int((N-2+N-row-1)*row/2)] = -1    ## あとで変更
                k = k+1
        return Phi
    
    def _create_Psi(self, N):
        Phi = self._create_Phi(N)   ## 入力にしちゃう
        tmp = torch.zeros(N, N**2)
        for row in range(N):
            tmp[row, row*(N+1)] = 1
        Psi = torch.mm(tmp, Phi)
        return Psi
    
    def _gsp_llap_pds(self, ell, X,
                      Phi, Psi,
                      alpha, beta, gamma):
        N = X.shape[0]
        vecXX_ = torch.t(torch.mm(X, torch.t(X))).contiguous().view(N**2,1)
        vecXX_ = self._min_max_normalization(vecXX_) * 10
        Phi_vecXX_ = torch.mm(Phi.T, vecXX_)
        Phi_Phi = torch.mm(Phi.T, Phi)
        theta = self.step_size_pds

        ## initialization
        b_0 = torch.mm(Psi, ell)

        for iter in range(self.iters_pds):
            z = ell - theta * (alpha * Phi_vecXX_ + 2 * gamma * ell + torch.mm(Psi.T, b_0))
            # z = ell - theta * (alpha * Phi_vecXX_ + gamma * torch.mm(Phi_Phi, ell) + torch.mm(Psi.T, b_0))
            z_0 = b_0 + theta * torch.mm(Psi, ell)
            p = torch.where(z>0, 0, z)
            p_0 = z_0 - theta * 0.5 * (z_0/theta + torch.sqrt((z_0/theta)**2 + 4 * beta / theta))
            q = p - theta * (alpha * Phi_vecXX_ + 2 * gamma * p + torch.mm(Psi.T, p_0))
            # q = p - theta * (alpha * Phi_vecXX_ + gamma * torch.mm(Phi_Phi, p) + torch.mm(Psi.T, p_0))
            q_0 = p_0 + theta * torch.mm(Psi, p)
            # print(torch.norm(-z_0 + q_0) / torch.norm(ell))
            if torch.norm(-z + q) / torch.norm(ell) < self.tol_pds:
                ell = ell - z + q
                break
            ell = ell - z + q
            b_0 = b_0 - z_0 + q_0
        return ell
    
    def forward(self, Y):
        """
        Input
        ------
        Y : torch.tensor
            N_s x N_m
        
        Output
        ------
        X : torch.tensor
            N_s x N_m
        """

        ## Get Size
        self.N_s = Y.shape[0]
        self.N_m = Y.shape[1]

        ## Initialization
        X_out_list = torch.zeros(self.layers+1, self.N_s, self.N_m)
        L_m_list = torch.zeros(self.layers+1, self.N_m, self.N_m)
        L_s_list = torch.zeros(self.layers+1, self.N_s, self.N_s)
        X_out_list[0,:,:] = Y
        L_m_list[0, :, :] = self.coefficient_L * (self.N_m*torch.eye(self.N_m) - torch.ones(self.N_m, self.N_m))
        L_s_list[0, :, :] = self.coefficient_L * (self.N_s*torch.eye(self.N_s) - torch.ones(self.N_s, self.N_s))

        ## loop
        X_out = X_out_list[0, :, :]
        L_m = L_m_list[0, :, :]
        L_s = L_s_list[0, :, :]
        for layer in range(self.layers):
            self.alpha_s[layer].data = self._relu(self.alpha_s[layer]).data
            self.beta_s[layer].data = self._relu(self.beta_s[layer]).data
            self.gamma_s[layer].data = self._relu(self.gamma_s[layer]).data
            self.alpha_m[layer].data = self._relu(self.alpha_m[layer]).data
            self.beta_m[layer].data = self._relu(self.beta_m[layer]).data
            self.gamma_m[layer].data = self._relu(self.gamma_m[layer]).data

            X_ = X_out.T
            # ell_m = self._gsp_llap_pds( ell=torch.mm(torch.linalg.pinv(self.Phi_m), L_m.T.reshape(N_m**2,1)), X=X_,
            #                             Phi=self.Phi_m, Psi=self.Psi_m,
            #                             alpha=self.alpha_m[layer], beta=self.beta_m[layer], gamma=self.gamma_m[layer])
            ell_m = self._gsp_llap_pds( ell=self._mat2vech(L_m), X=X_,
                                        Phi=self.Phi_m, Psi=self.Psi_m,
                                        alpha=self.alpha_m[layer], beta=self.beta_m[layer], gamma=self.gamma_m[layer])
            L_m = torch.mm(self.Phi_m, ell_m).reshape(self.N_m, self.N_m).T
            X_ = torch.mm( torch.linalg.pinv(torch.eye(self.N_m) + self.alpha_m[layer] * L_m), Y.T )
            # X_ = torch.linalg.solve((torch.eye(self.N_m) + self.alpha_m[layer] * L_m), Y.T)
            # ell_s = self._gsp_llap_pds( ell=torch.mm(torch.linalg.pinv(self.Phi_s), L_s.T.reshape(N_s**2,1)), X=X_.T,
            #                             Phi=self.Phi_s, Psi=self.Psi_s,
            #                             alpha=self.alpha_s[layer], beta=self.beta_s[layer], gamma=self.gamma_s[layer])
            ell_s = self._gsp_llap_pds( ell=self._mat2vech(L_s), X=X_.T,
                                        Phi=self.Phi_s, Psi=self.Psi_s,
                                        alpha=self.alpha_s[layer], beta=self.beta_s[layer], gamma=self.gamma_s[layer])
            L_s = torch.mm(self.Phi_s, ell_s).reshape(self.N_s, self.N_s).T
            X_out = torch.mm( torch.linalg.pinv(torch.eye(self.N_s) + self.alpha_s[layer] * L_s), X_.T )
            # X_out = torch.linalg.solve((torch.eye(self.N_s) + self.alpha_s[layer] * L_s), X_.T)

            X_out_list[layer+1, :, :] = X_out
            L_m_list[layer+1, :, :] = L_m
            L_s_list[layer+1, :, :] = L_s
        
        return L_m_list, L_s_list, X_out_list, X_out