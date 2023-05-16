import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from s4 import S4, S4D
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ResidualBlock(nn.Module):
    def __init__(self, num_tokens, **kwargs):
        super().__init__()
        
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

    def forward(self, x):
        # x(B, C, L)
        # diffusion_emb(B, embedding_dim)

        y = x
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

    def __init__(self, params, **kwargs) -> None:
        super().__init__()
        self.params = params
        in_chn = params.in_chn
        num_tokens = params.num_tokens
        depth = params.depth
        # dropout = params.dropout

        self.input_projection = nn.Conv1d(in_chn, num_tokens, 1)
        self.output_projection1 = nn.Conv1d(num_tokens, num_tokens, 1)
        self.output_projection2 = nn.Conv1d(num_tokens, 3, 1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(num_tokens, **kwargs)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        # x(B, L, K) -> (transposed == False), (B, K, L) -> (transposed == True)
        # t(B, 1, 1)
        x = self.input_projection(x) # (B, C, L)
        
        sum_skip = torch.zeros_like(x)
        for layer in self.residual_layers:
            x, skip = layer(x) # (B, C, L)
            sum_skip = sum_skip + skip
        
        x = sum_skip / math.sqrt(len(self.residual_layers))
        x = F.relu(self.output_projection1(x))
        x = self.output_projection2(x) 
        
        return x


def loss_fn(output, truth):
    loss = F.smooth_l1_loss(output, truth)
    return loss


def eval_error(output, truth):
    sum_err = np.sum((output-truth) ** 2, axis=(1,2))
    mean_err = np.mean(sum_err)
    return mean_err, sum_err


def align(data):
    if isinstance(data, torch.Tensor):
        return data.to(torch.float32)
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        return data.to(torch.float32)
    else: raise
    