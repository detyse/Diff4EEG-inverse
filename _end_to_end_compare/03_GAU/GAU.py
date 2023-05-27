import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pos_embs import RoPE
from einops.layers.torch import Rearrange
import math

class GAUBlock(nn.Module):
    def __init__(self, dim, hidden_dim=16, query_key_dim=64):
        super().__init__()
        self.transpose = Rearrange('b c l -> b l c')

        self.norm = nn.LayerNorm(dim)
        # self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU(),
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU(),
        )

        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)

        self.pos_emb = RoPE(query_key_dim)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
        )

        self.output_projection = nn.Conv1d(dim, dim*2, 1)

    def forward(self, x):
        x_transpose = self.transpose(x)

        seq_len = x_transpose.shape[-2]

        normed_x = self.norm(x_transpose)
        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)

        Z = self.to_qk(normed_x)

        QK = torch.einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)
        q = self.pos_emb(q)
        k = self.pos_emb(k)

        sim = torch.einsum('b i d, b j d -> b i j', q, k) / seq_len

        A = F.softmax(sim, dim=-1)
        # A = self.dropout(A)

        V = torch.einsum('b i j, b j d -> b i d', A ,v)
        O = u * V   # 表示哈达玛积

        out = self.to_out(O)
        out = self.transpose(out)
        out = self.output_projection(out)

        residual, skip = torch.chunk(out, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class GAUNet(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(1, params.dim, 1)
        self.residual_layers = nn.ModuleList([
            GAUBlock(dim=params.dim, hidden_dim=params.hidden_dim, query_key_dim=params.query_key_dim)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = nn.Conv1d(params.dim, params.dim, 1)
        self.output_projection = nn.Conv1d(params.dim, 3, 1)

    def forward(self, x):
        x = self.input_projection(x)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
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
