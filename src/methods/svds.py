import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

class SVDS():
  def __init__(self, threshold:float):
    self.threshold = threshold
  
  def forward(self, Y):
    self.N = Y.shape[1] # Nodes

    U,S,V = torch.svd(Y)
    S = torch.where(S > self.threshold, 0, S)
    S = torch.diag(S)
    X_out = torch.mm(U, torch.mm(S, V.T))

    return X_out