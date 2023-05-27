import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
import math

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):   # x is the time
        x_porj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_porj), torch.cos(x_porj)], dim = -1)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None]

class GAUBlock(nn.Module):
    def __init__(self, dim, hidden_dim=16, query_key_dim=64):
        super().__init__()
        self.embed_projection = Dense(dim * 3, dim)

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

    def forward(self, x, t):
        # time embedding
        embedding = self.embed_projection(t)
        x = x + embedding

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
    def __init__(self, params, marginal_prob_std):
        super().__init__()
        self.params = params
        self.marginal_prob_std = marginal_prob_std
        self.input_projection = nn.Conv1d(1, params.dim, 1)
        self.embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=params.embed_dim),
            nn.Linear(params.embed_dim, params.embed_dim)
        )
        self.embedding_layer = nn.Linear(params.embed_dim, params.dim * 3)
        self.residual_layers = nn.ModuleList([
            GAUBlock(dim=params.dim, hidden_dim=params.hidden_dim, query_key_dim=params.query_key_dim)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = nn.Conv1d(params.dim, params.dim, 1)
        self.output_projection = nn.Conv1d(params.dim, 1, 1)

    def forward(self, x, t):
        x = self.input_projection(x)
        x = F.silu(x)

        embed = self.embedding(t)
        embed = F.silu(embed)
        embed = self.embedding_layer(embed)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, embed)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x) / self.marginal_prob_std(t)[:, None, None]
        return x

class RoPE(nn.Module):
    def __init__(self, dim: int, max_lens: int = 5000):
        super().__init__()
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2)[: (dim//2)].float() / dim))
        t = torch.arange(max_lens, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        self.register_buffer('freqs_cis', freqs_cis)

    def reshape_for_broadcast(self, freqs_cis, x):
        freqs_cis = freqs_cis[:x.shape[1], :]
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])

        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    
    def forward(self, x):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # print(type(self.freqs_cis))
        freqs_cis = self.reshape_for_broadcast(self.freqs_cis, x_)
        x_out = torch.view_as_real(x_ * freqs_cis).flatten(2)
        return x_out.type_as(x)
    
class DW4BSS_GAU(nn.Module):
    def __init__(self, params, marginal_prob_std):
        super().__init__()
        self.DW1 = GAUNet(params, marginal_prob_std)
        self.DW2 = GAUNet(params, marginal_prob_std)
        self.DW3 = GAUNet(params, marginal_prob_std)
        
    def forward(self, x, t):
        x_0 = x[:, 0, :]
        x_0 = x_0[:, None, :]
        x_0 = self.DW1(x_0, t)
        
        x_1 = x[:, 1, :]
        x_1 = x_1[:, None, :]
        x_1 = self.DW2(x_1, t)
        
        x_2 = x[:, 2, :]
        x_2 = x_2[:, None, :]
        x_2 = self.DW3(x_2, t)
        return torch.cat([x_0, x_1, x_2], dim=1)
    
class GAUnet3(nn.Module):
    def __init__(self, params, marginal_prob_std):
        super().__init__()
        self.params = params
        self.marginal_prob_std = marginal_prob_std
        self.input_projection = nn.Conv1d(3, params.dim, 1)
        self.embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=params.embed_dim),
            nn.Linear(params.embed_dim, params.embed_dim)
        )
        self.embedding_layer = nn.Linear(params.embed_dim, params.dim * 3)
        self.residual_layers = nn.ModuleList([
            GAUBlock(dim=params.dim, hidden_dim=params.hidden_dim, query_key_dim=params.query_key_dim)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = nn.Conv1d(params.dim, params.dim, 1)
        self.output_projection = nn.Conv1d(params.dim, 3, 1)

    def forward(self, x, t):
        x = self.input_projection(x)
        x = F.silu(x)

        embed = self.embedding(t)
        embed = F.silu(embed)
        embed = self.embedding_layer(embed)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, embed)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x) / self.marginal_prob_std(t)[:, None, None]
        return x