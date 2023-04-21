import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:1')


#Encoder Architecture
#Input Dim : Num_Words
#Ouput Dim : Embedding Dim
#No of Layers : 1

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        x = self.lin(x)
        return x
    
#Decoder Architecture
#Input Dim : Embedding Dim
#Ouput Dim : Num_Words
#No of Layers : 1
#Softmax Output Function

class Decoder(nn.Module):
    def __init__(self, emb_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(emb_dim, output_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.softmax(x, dim=-1)
        return x


#Skip_Gram Model
#Adding encoder , decoder
class Skip_Gram(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, emb_dim).to(device)
        self.decoder = Decoder(emb_dim, input_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_embeddings(self, x):
        return self.encoder(x)

