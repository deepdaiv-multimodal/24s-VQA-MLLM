import math
import torch
import torch.nn.functional as F
from torch import nn

def rotate_every_two(x):
    x = x.view(x.shape[:-1] + (x.shape[-1] // 2, 2))
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).view(x.shape[:-2] + (x.shape[-2] * 2,))

def apply_rotary_pos_emb(q, k, sinu_pos):
    sin, cos = sinu_pos
    q_len, q_dim = q.shape[-2:]
    k_len, k_dim = k.shape[-2:]

    sin_q = sin[:q_len, :q_dim].view(1, 1, q_len, q_dim).expand(q.shape[0], q.shape[1], q_len, q_dim)
    cos_q = cos[:q_len, :q_dim].view(1, 1, q_len, q_dim).expand(q.shape[0], q.shape[1], q_len, q_dim)
    
    sin_k = sin[:k_len, :k_dim].view(1, 1, k_len, k_dim).expand(k.shape[0], k.shape[1], k_len, k_dim)
    cos_k = cos[:k_len, :k_dim].view(1, 1, k_len, k_dim).expand(k.shape[0], k.shape[1], k_len, k_dim)

    q = (q * cos_q) + (rotate_every_two(q) * sin_q)
    k = (k * cos_k) + (rotate_every_two(k) * sin_k)
    
    return q, k

class RoPE2d(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j->ij', pos, inv_freq)
        self.register_buffer('sin', sinusoid_inp.sin().unsqueeze(-1).expand(-1, -1, 2).reshape(sinusoid_inp.shape[0], -1))
        self.register_buffer('cos', sinusoid_inp.cos().unsqueeze(-1).expand(-1, -1, 2).reshape(sinusoid_inp.shape[0], -1))

    def forward(self, q, k):
        return apply_rotary_pos_emb(q, k, (self.sin, self.cos))