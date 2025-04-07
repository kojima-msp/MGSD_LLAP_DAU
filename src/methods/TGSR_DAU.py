import random
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import eigh

def soft_threshold(X, thr):
    """Soft-thresholding operator."""
    return F.relu(X - thr) - F.relu(-X - thr)


class TGSR_DAU(nn.Module):
    """Graph signal matrix denoising"""
    def __init__(self, layers=1, default: Optional[float]=0.05):
        super().__init__()
        if default is None:
            default = random.random()
        self.mr1 = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)]).double()
        self.mr2 = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)]).double()
        self.mr3 = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)]).double()
        self.mc1 = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)]).double()
        self.mc2 = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)]).double()
        self.mc3 = nn.ParameterList([nn.Parameter(torch.tensor([default])) for _ in range(layers)]).double()
        self.layers = layers 
    
    ## ReLU
    def _relu(self, x):
        return torch.maximum(x, torch.tensor(1e-08))

    def forward(self, Y):
        
        Mr, Mc, Ur, Uc, Lamd_r, Lamd_c = self.preprocess(Y)

        Y = Y.double()
        X = Y.clone()
        Ir = torch.eye(Y.shape[0])
        Ic = torch.eye(Y.shape[1])
        for i in range(self.layers):
            self.mr1[i].data = self._relu(self.mr1[i]).data
            self.mr2[i].data = self._relu(self.mr2[i]).data
            self.mr3[i].data = self._relu(self.mr3[i]).data
            self.mc1[i].data = self._relu(self.mc1[i]).data
            self.mc2[i].data = self._relu(self.mc2[i]).data
            self.mc3[i].data = self._relu(self.mc3[i]).data

            X_ = X.T

            Zc = soft_threshold(torch.mm(Mc, X_), self.mc1[i])
            Zr = soft_threshold(torch.mm(Mr, X), self.mr1[i])

            Xc1 = torch.mm(Uc, torch.inverse(Ic + self.mc2[i]*Lamd_c))
            Xc2 = torch.mm(Xc1, Uc.T)
            tmp_c = Y.T + self.mc3[i] * torch.mm(Mc.transpose(1, 0), Zc)
            X_ = torch.mm(Xc2, tmp_c.double())

            Xr1 = torch.mm(Ur, torch.inverse(Ir + self.mr2[i]*Lamd_r))
            Xr2 = torch.mm(Xr1, Ur.T)
            tmp_r = X_.T + self.mr3[i] * torch.mm(Mr.transpose(1, 0), Zr)
            X = torch.matmul(Xr2, tmp_r.double())
            
        return X

    def preprocess(self, Y):
        N_s, N_m = Y.shape

        # RBF
        Z_s = torch.cdist(Y, Y) ** 2
        Z_m = torch.cdist(Y.T, Y.T) ** 2
        Z_s = (Z_s - torch.min(Z_s)) / (torch.max(Z_s) - torch.min(Z_s))
        Z_m = (Z_m - torch.min(Z_m)) / (torch.max(Z_m) - torch.min(Z_m))
        Z_s = Z_s * 10
        Z_m = Z_m * 10
        # Z_s = Z_s / (M * K) ## 正規化の代わりに平均化
        # Z_m = Z_m / (N * K) ## 正規化の代わりに平均化
        W_s = torch.mul(torch.exp(-Z_s / torch.tensor(1)), torch.ones((N_s, N_s))-torch.eye(N_s))
        W_m = torch.mul(torch.exp(-Z_m / torch.tensor(1)), torch.ones((N_m, N_m))-torch.eye(N_m))
        D_s = torch.squeeze(W_s @ torch.ones(N_s,1))
        D_m = torch.squeeze(W_m @ torch.ones(N_m,1))
        L_s = torch.diag(D_s) - W_s
        L_m = torch.diag(D_m) - W_m
        L_s_norm = torch.mm(torch.diag(torch.pow(D_s, -0.5)), torch.mm(L_s, torch.diag(torch.pow(D_s, -0.5))))
        L_m_norm = torch.mm(torch.diag(torch.pow(D_m, -0.5)), torch.mm(L_m, torch.diag(torch.pow(D_m, -0.5))))
        W_s_norm = torch.mm(torch.diag(torch.pow(D_s, -0.5)), torch.mm(W_s, torch.diag(torch.pow(D_s, -0.5))))
        W_m_norm = torch.mm(torch.diag(torch.pow(D_m, -0.5)), torch.mm(W_m, torch.diag(torch.pow(D_m, -0.5))))
        M_s = torch.zeros((N_s*N_s, N_s))
        M_m = torch.zeros((N_m*N_m, N_m))
        for s in range(N_s*N_s):
            i = int(s/N_s)
            for t in range(N_s):
                j = s - i*N_s
                M_s[s, i] = torch.sqrt(W_s_norm[i, j])
                M_s[s, j] = -torch.sqrt(W_s_norm[i, j])
        for s in range(N_m*N_m):
            i = int(s/N_m)
            for t in range(N_m):
                j = s - i*N_m
                M_m[s, i] = torch.sqrt(W_m_norm[i, j])
                M_m[s, j] = -torch.sqrt(W_m_norm[i, j])
        Lamb_s, U_s = eigh(L_s_norm)
        Lamb_m, U_m = eigh(L_m_norm)

        return M_s, M_m, U_s, U_m, torch.diag(Lamb_s), torch.diag(Lamb_m)