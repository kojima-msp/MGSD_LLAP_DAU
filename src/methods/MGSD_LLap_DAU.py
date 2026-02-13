"""
Algorithm Unrolling-based Denoising of Multimodal Graph Signals (MGSD_LLap_DAU) method.
"""
"""
@Author: Hayate Kojima
@Contact: h-kojima@msp-lab.org
@Date: 2025/04/01
"""

import torch
from torch import nn
from typing import Optional


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)



class MGSD_LLap_DAU(nn.Module):
    def __init__(self, layers:int, N_s:int, N_m:int, default:Optional[float]=0.5, step_size_pds:float=5e-3, iters_pds:int=1000, tol_pds:float=1e-3):
        """
        Input
        ------
        layers : int [Number of layers]
        N_s : int [Number of spatial nodes]
        N_m : int [Number of modality nodes]
        default : float [(Optional) Initial value of learnable parameters]
        step_size_pds : float [(Optional) Step size of PDS algorithm]
        iters_pds : int [(Optional) Number of iterations of PDS algorithm]
        tol_pds : float [(Optional) Tolerance of PDS algorithm]
        """

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
        self.N_m = N_m
        self.N_s = N_s
        self.Phi_m = self._create_Phi(N_m)
        self.Phi_s = self._create_Phi(N_s)
        self.Psi_m = self._create_Psi(N_m)
        self.Psi_s = self._create_Psi(N_s)
        
        # added for ablation study
        self.L_s = None
        self.L_m = None
    
        ## ReLU
        self._relu = nn.ReLU()
    
    
    def _min_max_normalization(self, x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    ## convert matrix to half-vector
    ## edited by Hayate Kojima (2026/02/13)
    def _mat2vech(self, L):
        N = L.shape[0]
        rows, cols = torch.triu_indices(N, N, offset=1)
        return L[rows, cols].unsqueeze(1)
    
    ## arange transformation matrix that convert vec(L) from vech(L)
    ## edited by Hayate Kojima (2026/02/13)
    def _create_Phi(self, N):
        M = int(N * (N - 1) / 2)
        Phi = torch.zeros(N**2, M)

        rows, cols = torch.tril_indices(N, N, offset=-1)
        sort_order = torch.argsort(cols * N + rows)
        rows = rows[sort_order]
        cols = cols[sort_order]

        vech_indices = torch.arange(M)

        # --- Lower Triangle (row > col) ---
        k_lower = cols * N + rows
        Phi[k_lower, vech_indices] = 1.0

        diag_indices_lower = cols * (N + 1)
        Phi[diag_indices_lower, vech_indices] = -1.0

        # --- Upper Triangle (row < col) ---
        k_upper = rows * N + cols
        Phi[k_upper, vech_indices] = 1.0

        diag_indices_upper = rows * (N + 1)
        Phi[diag_indices_upper, vech_indices] = -1.0

        return Phi

    ## edited by Hayate Kojima (2026/02/13)
    def _create_Psi(self, N):

        Phi = self._create_Phi(N)
        diag_indices = torch.arange(0, N**2, N + 1)
        Psi = Phi[diag_indices]
        
        return Psi
    
    def _gsp_llap_pds(self, ell, X,
                      Phi, Psi,
                      alpha, beta, gamma):
        N = X.shape[0]
        vecXX_ = torch.t(torch.mm(X, torch.t(X))).contiguous().view(N**2,1)
        vecXX_ = self._min_max_normalization(vecXX_) * 10
        Phi_vecXX_ = torch.mm(Phi.T, vecXX_)
        theta = self.step_size_pds

        ## initialization
        b_0 = torch.mm(Psi, ell)

        for iter in range(self.iters_pds):
            z = ell - theta * (alpha * Phi_vecXX_ + 2 * gamma * ell + torch.mm(Psi.T, b_0))
            z_0 = b_0 + theta * torch.mm(Psi, ell)
            p = torch.where(z>0, 0, z)
            p_0 = z_0 - theta * 0.5 * (z_0/theta + torch.sqrt((z_0/theta)**2 + 4 * beta / theta))
            q = p - theta * (alpha * Phi_vecXX_ + 2 * gamma * p + torch.mm(Psi.T, p_0))
            q_0 = p_0 + theta * torch.mm(Psi, p)
            if torch.norm(-z + q) / torch.norm(ell) < self.tol_pds:
                ell = ell - z + q
                break
            ell = ell - z + q
            b_0 = b_0 - z_0 + q_0
        return ell
    
    def forward(self, Y, L_m:Optional[torch.tensor]=None, L_s:Optional[torch.tensor]=None):
        """
        Input
        ------
        Y : torch.tensor [Observed signals]
            N_s x N_m
        L_m : torch.tensor [(Optional) Groundtruth graph Laplacian for modality m (only used in ablation study)]
            N_m x N_m
        L_s : torch.tensor [(Optional) Groundtruth graph Laplacian for modality s (only used in ablation study)]
            N_s x N_s
        
        Output
        ------
        X : torch.tensor [Estimated denoised signals]
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
        L_m_list[0, :, :] = 0.1 * (self.N_m*torch.eye(self.N_m) - torch.ones(self.N_m, self.N_m)) # fully connected graph
        L_s_list[0, :, :] = 0.1 * (self.N_s*torch.eye(self.N_s) - torch.ones(self.N_s, self.N_s)) # fully connected graph

        ## initialize L_m and L_s for ablation study
        self.L_m = L_m
        self.L_s = L_s

        X_out = X_out_list[0, :, :]
        L_m = L_m_list[0, :, :] if self.L_m is None else self.L_m
        L_s = L_s_list[0, :, :] if self.L_s is None else self.L_s
        
        ## loop
        for layer in range(self.layers):
            alpha_s = self._relu(self.alpha_s[layer])
            alpha_m = self._relu(self.alpha_m[layer])
            beta_s = self._relu(self.beta_s[layer])
            beta_m = self._relu(self.beta_m[layer])
            gamma_s = self._relu(self.gamma_s[layer])
            gamma_m = self._relu(self.gamma_m[layer])

            X_ = X_out.T

            # line 2 in Algorithm 2
            if self.L_m is None:
                ell_m = self._gsp_llap_pds( ell=self._mat2vech(L_m), X=X_,
                                        Phi=self.Phi_m, Psi=self.Psi_m,
                                        alpha=alpha_m, beta=beta_m, gamma=gamma_m)
                L_m = torch.mm(self.Phi_m, ell_m).reshape(self.N_m, self.N_m).T
            else: # use groundtruth/Learned L_m
                L_m = self.L_m
            
            # line 3 in Algorithm 2
            X_ = torch.mm(torch.linalg.pinv(torch.eye(self.N_m) + alpha_m * L_m), Y.T)

            # line 4 in Algorithm 2
            if self.L_s is None:
                ell_s = self._gsp_llap_pds( ell=self._mat2vech(L_s), X=X_.T,
                                        Phi=self.Phi_s, Psi=self.Psi_s,
                                        alpha=alpha_s, beta=beta_s, gamma=gamma_s)
                L_s = torch.mm(self.Phi_s, ell_s).reshape(self.N_s, self.N_s).T
            else: # use groundtruth/Learned L_s
                L_s = self.L_s
            
            # line 5 in Algorithm 2
            X_out = torch.mm(torch.linalg.pinv(torch.eye(self.N_s) + alpha_s * L_s), X_.T)

            X_out_list[layer+1, :, :] = X_out
            L_m_list[layer+1, :, :] = L_m
            L_s_list[layer+1, :, :] = L_s
        
        return L_m_list, L_s_list, X_out_list, X_out