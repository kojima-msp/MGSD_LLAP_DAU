'''
    Traditional autoencoder (AE)
    1. Encoder: y = s(Wx + b)
                s() is the sigmoid function
    2. Decoder: y = Wx+b
                For real valued x, due to the Gaussian interpretation, 
                it is more natural not to use a squashing nonlinearrity in the decoder.
    see: "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion."
'''
import torch
import torch.nn as nn



#%%
## y = s(Wx+b), and s() is the sigmoid function
class Encoder01(nn.Module):
    def __init__(self, input_size, BIAS):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.enc1 = torch.nn.Linear(input_size, BIAS)
    def forward(self, x):
        x = self.sigmoid(self.enc1(x))
        return x
class Encoder03(nn.Module):
    def __init__(self, input_size, BIAS):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.enc1 = torch.nn.Linear(input_size, BIAS*2)
        self.enc2 = torch.nn.Linear(BIAS*2, BIAS)
    def forward(self, x):
        x = self.sigmoid(self.enc1(x))
        x = self.sigmoid(self.enc2(x))
        return x
class Encoder05(nn.Module):
    def __init__(self, input_size, BIAS):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.enc1 = torch.nn.Linear(input_size, BIAS*3)
        self.enc2 = torch.nn.Linear(BIAS*3, BIAS*2)
        self.enc3 = torch.nn.Linear(BIAS*2, BIAS)
    def forward(self, x):
        x = self.sigmoid(self.enc1(x))
        x = self.sigmoid(self.enc2(x))
        x = self.sigmoid(self.enc3(x))
        return x
class Encoder07(nn.Module):
    def __init__(self, input_size, BIAS):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.enc1 = torch.nn.Linear(input_size, BIAS*4)
        self.enc2 = torch.nn.Linear(BIAS*4, BIAS*3)
        self.enc3 = torch.nn.Linear(BIAS*3, BIAS*2)
        self.enc4 = torch.nn.Linear(BIAS*2, BIAS)
    def forward(self, x):
        x = self.sigmoid(self.enc1(x))
        x = self.sigmoid(self.enc2(x))
        x = self.sigmoid(self.enc3(x))
        x = self.sigmoid(self.enc4(x))
        return x
class Encoder09(nn.Module):
    def __init__(self, input_size, BIAS):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.enc1 = torch.nn.Linear(input_size, BIAS*5)
        self.enc2 = torch.nn.Linear(BIAS*5, BIAS*4)
        self.enc3 = torch.nn.Linear(BIAS*4, BIAS*3)
        self.enc4 = torch.nn.Linear(BIAS*3, BIAS*2)
        self.enc5 = torch.nn.Linear(BIAS*2, BIAS)
    def forward(self, x):
        x = self.sigmoid(self.enc1(x))
        x = self.sigmoid(self.enc2(x))
        x = self.sigmoid(self.enc3(x))
        x = self.sigmoid(self.enc4(x))
        x = self.sigmoid(self.enc5(x))
        return x

#%%
## y = Wx+b
class Decoder01(nn.Module):
    def __init__(self, output_size, BIAS):
        super().__init__()
        self.dec1 = torch.nn.Linear(BIAS, output_size)
    def forward(self, x):
        x = self.dec1(x)
        return x
class Decoder03(nn.Module):
    def __init__(self, output_size, BIAS):
        super().__init__()
        self.dec1 = torch.nn.Linear(BIAS, BIAS*2)
        self.dec2 = torch.nn.Linear(BIAS*2, output_size)
    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        return x
class Decoder05(nn.Module):
    def __init__(self, output_size, BIAS):
        super().__init__()
        self.dec1 = torch.nn.Linear(BIAS, BIAS*2)
        self.dec2 = torch.nn.Linear(BIAS*2, BIAS*3)
        self.dec3 = torch.nn.Linear(BIAS*3, output_size)
    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x
class Decoder07(nn.Module):
    def __init__(self, output_size, BIAS):
        super().__init__()
        self.dec1 = torch.nn.Linear(BIAS, BIAS*2)
        self.dec2 = torch.nn.Linear(BIAS*2, BIAS*3)
        self.dec3 = torch.nn.Linear(BIAS*3, BIAS*4)
        self.dec4 = torch.nn.Linear(BIAS*4, output_size)
    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x
class Decoder09(nn.Module):
    def __init__(self, output_size, BIAS):
        super().__init__()
        self.dec1 = torch.nn.Linear(BIAS, BIAS*2)
        self.dec2 = torch.nn.Linear(BIAS*2, BIAS*3)
        self.dec3 = torch.nn.Linear(BIAS*3, BIAS*4)
        self.dec4 = torch.nn.Linear(BIAS*4, BIAS*5)
        self.dec5 = torch.nn.Linear(BIAS*5, output_size)
    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        return x

#%%
## main
class AutoEncoder(nn.Module):
    def __init__(self, oug_size, N_layers, BIAS):
        super().__init__()
        if N_layers == 1:
            self.encoder = Encoder01(oug_size, BIAS)
            self.decoder = Decoder01(oug_size, BIAS)
        elif N_layers == 3:
            self.encoder = Encoder03(oug_size, BIAS)
            self.decoder = Decoder03(oug_size, BIAS)
        elif N_layers == 5:
            self.encoder = Encoder05(oug_size, BIAS)
            self.decoder = Decoder05(oug_size, BIAS)
        elif N_layers == 7:
            self.encoder = Encoder07(oug_size, BIAS)
            self.decoder = Decoder07(oug_size, BIAS)
        elif N_layers == 9:
            self.encoder = Encoder09(oug_size, BIAS)
            self.decoder = Decoder09(oug_size, BIAS)
    def forward(self, X):
        N_s, N_m = X.shape
        x = X.reshape(N_s*N_m)
        x = self.encoder(x) # encoder
        x = self.decoder(x) # decoder
        X = x.reshape(N_s, N_m)
        return X
