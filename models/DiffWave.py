from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import sqrt

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):   # x is the time
        x_porj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_porj), torch.cos(x_porj)], dim = -1)


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels, 2 * residual_channels, 3,
            padding = dilation, dilation = dilation)
        self.norm = nn.BatchNorm1d(2 * residual_channels)
        self.embed_projection = nn.Linear(residual_channels * 3, residual_channels)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, t):
        embedding = self.embed_projection(t)[..., None]
        y = x + embedding
        y = self.dilated_conv(y)
        # 加一个norm层
        y = self.norm(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class DiffWave(nn.Module):
    '''
    ref: 
    '''
    def __init__(self, nscheduler, config, **kwargs):
        super().__init__()
        self.config = config
        self.nscheduler = nscheduler
        self.input_projection = Conv1d(3, config.diffwave.residual_channels, 1)
        # 这里
        self.embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=config.diffwave.embed_dim),
            nn.Linear(config.diffwave.embed_dim, config.diffwave.embed_dim)
        )
        self.embedding_layer = nn.Linear(config.diffwave.embed_dim, config.diffwave.residual_channels * 3)
        
        self.residual_layers = nn.ModuleList([
            ResidualBlock(config.diffwave.residual_channels, 
            2**(i % config.diffwave.dilation_cycle_length))
            for i in range(config.diffwave.residual_layers)
        ])
        self.skip_projection = Conv1d(config.diffwave.residual_channels, config.diffwave.residual_channels, 1)
        self.output_projection = Conv1d(config.diffwave.residual_channels, 3, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, t):
        # [32, 512]
        # x = x.unsqueeze(1)  # [32, 1, 512]
        x = self.input_projection(x)
        x = F.relu(x)

        embed = self.embedding(t.view(-1))
        embed = silu(embed)
        embed = self.embedding_layer(embed)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, embed)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        x = x / torch.sqrt(self.nscheduler.BETA(t))
        return x


# class DW4BSS(nn.Module):
#     def __init__(self, config, marginal_prob_std):
#         super().__init__()
#         self.DW1 = DiffWave(config, marginal_prob_std)
#         self.DW2 = DiffWave(config, marginal_prob_std)
#         self.DW3 = DiffWave(config, marginal_prob_std)
        
#     def forward(self, x, t):
#         x_0 = x[:, 0, :]
#         x_0 = x_0[:, None, :]
#         x_0 = self.DW1(x_0, t)
        
#         x_1 = x[:, 1, :]
#         x_1 = x_1[:, None, :]
#         x_1 = self.DW2(x_1, t)
        
#         x_2 = x[:, 2, :]
#         x_2 = x_2[:, None, :]
#         x_2 = self.DW3(x_2, t)
#         return torch.cat([x_0, x_1, x_2], dim=1)
    