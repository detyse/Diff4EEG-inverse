import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.s4 import S4, S4D
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):   # x is the time
        x_porj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_porj), torch.cos(x_porj)], dim = -1)


class ResidualBlock(nn.Module):

    def __init__(self, num_tokens, embedding_dim, **kwargs):
        super().__init__()
    
        self.diffusion_projection = nn.Sequential(
            nn.Linear(embedding_dim, num_tokens),
            Rearrange('b c -> b c 1')
        )
        
        self.input_projection = nn.Conv1d(num_tokens, 2 * num_tokens, 1)
        self.output_projection = nn.Conv1d(num_tokens, 2 * num_tokens, 1)
        
        self.S4_1 = S4(2 * num_tokens, **kwargs)
        self.ln1 = nn.Sequential(
            Rearrange('b c l -> b l c'),
            nn.LayerNorm(2 * num_tokens),
            Rearrange('b l c -> b c l'),
        )
        self.S4_2 = S4(2 * num_tokens, **kwargs)
        self.ln2 = nn.Sequential(
            Rearrange('b c l -> b l c'),
            nn.LayerNorm(2 * num_tokens),
            Rearrange('b l c -> b c l'),
        )

    def forward(self, x, diffusion_emb):
        # x(B, C, L)
        # diffusion_emb(B, embedding_dim)

        diffusion_emb = self.diffusion_projection(diffusion_emb) # (B, C, 1)
        y = x + diffusion_emb
        y = self.input_projection(y) # (B, 2C, L)
        y, _ = self.S4_1(y) # (B, 2C, L)
        y = self.ln1(y)
        y, _ = self.S4_2(y) # (B, 2C, L)
        y = self.ln2(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B, C, L)
        y = self.output_projection(y) # (B, 2C, L)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class SSSD(nn.Module):
    '''
    ref: Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models
    SSSD net without conditional embedding
    '''
    # def __init__(self, nscheduler, in_chn, num_tokens, depth, dropout=0, embedding_dim=128, transposed=False, **kwargs):

    def __init__(self, nscheduler, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.nscheduler = nscheduler
        in_chn = config.net.in_chn
        num_tokens = config.net.num_tokens
        depth = config.net.depth
        # dropout = config.dropout
        embedding_dim = config.net.embedding_dim

        self.diffusion_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU()
        )

        self.input_projection = nn.Conv1d(in_chn, num_tokens, 1)
        self.output_projection1 = nn.Conv1d(num_tokens, num_tokens, 1)
        self.output_projection2 = nn.Conv1d(num_tokens, in_chn, 1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(num_tokens, embedding_dim, **kwargs)
                for _ in range(depth)
            ]
        )

        self.nscheduler = nscheduler

    def forward(self, x, t):
        # x(B, L, K) -> (transposed == False), (B, K, L) -> (transposed == True)
        # t(B, 1, 1)
        diffusion_emb = self.diffusion_embedding(t.view(-1)) # (B, embedding_dim)
        x = self.input_projection(x) # (B, C, L)

        sum_skip = torch.zeros_like(x)
        for layer in self.residual_layers:
            x, skip = layer(x, diffusion_emb) # (B, C, L)
            sum_skip = sum_skip + skip
        
        x = sum_skip / math.sqrt(len(self.residual_layers))
        x = F.relu(self.output_projection1(x))
        x = self.output_projection2(x) 

        x = x / torch.sqrt(self.nscheduler.BETA(t)) # ?
        return x