import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GAU():
    def __init__(self, *args, **kwargs) -> None:
        hidden_dim = int(expansion_factor * dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

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

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.add_residual = add_residual

    def forward(self, x):
        seq_len = x.shape[-2]

        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)

        z = self.to_qk(normed_x)

        QK = torch.einsum('...d, h d -> b i j', q, k) / seq_len

        A = F.relu(sim)**2
        A = self.dropout(A)

        V = torch.einsum('b i j, b j d -> b i d', A ,v)
        V = v * gate

        out = self.to_out(V)

        if self.add_residual:
            out = out + x

        return out
