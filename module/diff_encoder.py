import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from module.transformer.Modules import ScaledDotProductAttention
import torch
from module.transformer.SubLayers import PositionwiseFeedForward


class DiffGuiderEncoder(nn.Module):
    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_modals, dropout=0.1, scale_emb=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DiffAllEncoderLayer(d_model, d_inner, n_head, d_k, d_v, n_modals, dropout=dropout)
            for _ in range(n_layers)])
        self.muti_layer_norm = MutiLayerNorm(d_model, n_modals, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, rep, modal_len_list, src_mask, return_attns=False):   #src_seq: (batch_size, seq_len)

        enc_slf_attn_list = []

        # -- Forward
        if self.scale_emb:
            rep *= self.d_model ** 0.5
        rep = self.dropout(rep)
        rep = self.muti_layer_norm(rep, modal_len_list)

        for enc_layer in self.layer_stack:
            rep, enc_slf_attn = enc_layer(rep, modal_len_list, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return rep, enc_slf_attn_list
        return rep,


class DiffAllEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, n_modals, dropout=0.1):
        super().__init__()
        self.slf_attn = DiffLNModalMultiHeadAttention(n_head, d_model, d_k, d_v, n_modals, dropout=dropout)
        self.pos_ffn = MutiPositionwiseFeedForward(d_model, d_inner, n_modals, dropout=dropout)

    def forward(self, muti_modal_rep, modal_len_list, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            muti_modal_rep, modal_len_list, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output, modal_len_list)
        return enc_output, enc_slf_attn


class DiffLNModalMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, n_modals, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qkv_list = nn.ModuleList([DiffQKV(n_head, d_model, d_k, d_v) for _ in range(n_modals)])
        self.fc_list = nn.ModuleList([nn.Linear(n_head * d_v, d_model) for _ in range(n_modals)])

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

        self.muti_layer_norm = MutiLayerNorm(d_model, n_modals, eps=1e-6)

    def forward(self, muti_modal_rep, modal_len_list, mask=None):

        sz_b, len_q, len_k, len_v = muti_modal_rep.size(0), muti_modal_rep.size(1),\
                                    muti_modal_rep.size(1), muti_modal_rep.size(1)

        residual = muti_modal_rep

        modal_rep_list = torch.split(muti_modal_rep, split_size_or_sections=modal_len_list, dim=1)
        qkv_list = []
        for i, w_qkv in enumerate(self.w_qkv_list):
            qkv_list.append(w_qkv(modal_rep_list[i]))

        q, k, v = zip(*qkv_list)

        q = torch.cat(q, dim=1)
        k = torch.cat(k, dim=1)
        v = torch.cat(v, dim=1)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        muti_q_list = torch.split(q, split_size_or_sections=modal_len_list, dim=1)
        result_q_list = []
        for i, fc in enumerate(self.fc_list):
            result_q_list.append(fc(muti_q_list[i]))
        q = torch.cat(result_q_list, dim=1)

        q = self.dropout(q)
        q += residual

        q = self.muti_layer_norm(q, modal_len_list)

        return q, attn


class DiffQKV(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

    def forward(self, modal_rep):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = modal_rep.size(0), modal_rep.size(1), modal_rep.size(1), modal_rep.size(1)

        q = self.w_qs(modal_rep).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(modal_rep).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(modal_rep).view(sz_b, len_v, n_head, d_v)

        return q, k, v


class MutiPositionwiseFeedForward(nn.Module):
    def __init__(self,  d_in, d_hid, n_modals, dropout=0.1):
        super().__init__()
        self.pos_forward_list = nn.ModuleList([PositionwiseFeedForward(d_in, d_hid, dropout)
                                               for _ in range(n_modals)])

    def forward(self, muti_x, modal_len_list):
        muti_x_list = torch.split(muti_x, split_size_or_sections=modal_len_list, dim=1)
        x_result_list = []
        for i, pf in enumerate(self.pos_forward_list):
            x_result_list.append(pf(muti_x_list[i]))

        x = torch.cat(x_result_list, dim=1)

        return x


class MutiLayerNorm(nn.Module):
    def __init__(self, d_model, n_modals, eps):
        super().__init__()
        self.layer_norm_list = nn.ModuleList([nn.LayerNorm(d_model, eps=eps)
                                              for _ in range(n_modals)])

    def forward(self, muti_x, modal_len_list):
        muti_x_list = torch.split(muti_x, split_size_or_sections=modal_len_list, dim=1)
        x_result_list = []
        for i, ln in enumerate(self.layer_norm_list):
            x_result_list.append(ln(muti_x_list[i]))

        x = torch.cat(x_result_list, dim=1)
        return x

