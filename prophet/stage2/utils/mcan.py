import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHAtt(nn.Module):
    """ Multi-Head Attention """
    def __init__(self, hidden_size, num_heads):
        super(MHAtt, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, v, k, q):
        n_batches = q.size(0)
        d_k = self.hidden_size // self.num_heads
        v = self.linear_v(v).view(n_batches, -1, self.num_heads, d_k).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.num_heads, d_k).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.num_heads, d_k).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(n_batches, -1, self.hidden_size)
        out = self.linear_merge(out)
        out = self.norm(out + q.transpose(1, 2).contiguous().view(n_batches, -1, self.hidden_size))
        return out

class MCA(nn.Module):
    """ Modular Co-Attention Network """
    def __init__(self, hidden_size, num_heads, num_layers):
        super(MCA, self).__init__()
        self.layers = nn.ModuleList([MHAtt(hidden_size, num_heads) for _ in range(num_layers)])

    def forward(self, img_feat, ques_feat):
        for layer in self.layers:
            img_feat = layer(img_feat, img_feat, ques_feat)
            ques_feat = layer(ques_feat, ques_feat, img_feat)
        return img_feat, ques_feat
