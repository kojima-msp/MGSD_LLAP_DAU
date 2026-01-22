import torch
from torch.linalg import eigh

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

class GLPF():
  def __init__(self, tau:float):
    self.tau = tau
  
  def forward(self, Y):
    self.N_s = Y.shape[0]
    
    Z_s = torch.cdist(Y, Y) ** 2
    Z_s = (Z_s - torch.min(Z_s)) / (torch.max(Z_s) - torch.min(Z_s))
    Z_s = Z_s * 10
    W_s = torch.mul(torch.exp(-Z_s / torch.tensor(1)), torch.ones((self.N_s, self.N_s))-torch.eye(self.N_s))
    D_s = torch.squeeze(W_s @ torch.ones(self.N_s, 1))
    L_s = torch.diag(D_s) - W_s
    Lamb_s, U_s = eigh(L_s)

    filtered_lambda_s = torch.diag(torch.pow((1 + self.tau*Lamb_s), -1))
    X_out = U_s @ filtered_lambda_s @ U_s.T @ Y
    
    return X_out