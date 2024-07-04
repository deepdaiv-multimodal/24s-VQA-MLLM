from net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, config):
        super(MHAtt, self).__init__()
        self.config = config

        self.linear_v = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.linear_k = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.linear_q = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)

        self.dropout = nn.Dropout(config.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.config.MULTI_HEAD,
            self.config.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.config.MULTI_HEAD,
            self.config.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.config.MULTI_HEAD,
            self.config.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.config.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=config.HIDDEN_SIZE,
            mid_size=config.FF_SIZE,
            out_size=config.HIDDEN_SIZE,
            dropout_r=config.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, config):
        super(SA, self).__init__()

        self.mhatt = MHAtt(config)
        self.ffn = FFN(config)

        self.dropout1 = nn.Dropout(config.DROPOUT_R)
        self.norm1 = LayerNorm(config.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(config.DROPOUT_R)
        self.norm2 = LayerNorm(config.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, config):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(config)
        self.mhatt2 = MHAtt(config)
        self.ffn = FFN(config)

        self.dropout1 = nn.Dropout(config.DROPOUT_R)
        self.norm1 = LayerNorm(config.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(config.DROPOUT_R)
        self.norm2 = LayerNorm(config.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(config.DROPOUT_R)
        self.norm3 = LayerNorm(config.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, config):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(config) for _ in range(config.LAYER)])
        self.dec_list = nn.ModuleList([SGA(config) for _ in range(config.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
