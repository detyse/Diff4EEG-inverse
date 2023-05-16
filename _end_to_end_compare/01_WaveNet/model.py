import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import sqrt

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation) -> None:
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x):
        y = self.dilated_conv(x)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class WaveNet(nn.Module): 
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        # self.device = device
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.residual_channels, 2**(i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 3, 1)

    def forward(self, x):
        # x = x.to(self.device)
        x = self.input_projection(x)
        
        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x)
            skip = skip_connection if skip is None else skip_connection + skip
        
        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
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
