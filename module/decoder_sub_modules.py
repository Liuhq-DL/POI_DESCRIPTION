import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import model_config
from utils.utils_b import join_lists


class OnlyAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        return attn


class MutiModalsAttValue(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

    def forward(self, low_rep_sub_list):
        batch_size = low_rep_sub_list[0].size(0)
        low_len_list = [x.size(1) for x in low_rep_sub_list]

        low_k_list = []
        low_v_list = []
        for rep, low_len in zip(low_rep_sub_list, low_len_list):
            low_k = self.w_ks(rep).view(batch_size, low_len, self.n_head, self.d_k)
            low_v = self.w_vs(rep).view(batch_size, low_len, self.n_head, self.d_v)
            low_k_list.append(low_k)
            low_v_list.append(low_v)

        low_k_list = [x_k.transpose(1, 2) for x_k in low_k_list]
        low_v_list = [x_v.transpose(1, 2) for x_v in low_v_list]

        return low_k_list, low_v_list, low_len_list


class SelectorModule(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.modals_num = model_config["sub_module_config"]["modals_num"]

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.low_att_value_list = nn.ModuleList([MutiModalsAttValue(n_head, d_model, d_k, d_v)
                                                 for _ in range(self.modals_num)])
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.diff_guider_k = DecDiffK(n_head, d_model, d_k, n_modals=self.modals_num)

        self.only_attention = OnlyAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_rep, guider_rep, guider_mask, low_rep_list, low_mask_list):
        batch_size = trg_rep.size(0)
        trg_len = trg_rep.size(1)

        residual = trg_rep
        trg_rep = self.layer_norm(trg_rep)

        trg_q = self.w_qs(trg_rep).view(batch_size, trg_len, self.n_head, self.d_k)

        modals_num_list = [len(x_item) for x_item in low_rep_list]
        guider_k = self.diff_guider_k(guider_rep, modals_num_list)

        low_k_list = []
        low_v_list = []
        low_len_list = []

        for i, att_value in enumerate(self.low_att_value_list):
            sub_low_k_list, sub_low_v_list, sub_low_len_list = att_value(low_rep_list[i])
            low_k_list += sub_low_k_list
            low_v_list += sub_low_v_list
            low_len_list += sub_low_len_list

        trg_q = trg_q.transpose(1, 2)
        guider_k = guider_k.transpose(1, 2)

        guider_mask = guider_mask.unsqueeze(1)
        guider_att = self.only_attention(trg_q, guider_k, mask=guider_mask)

        low_mask_list = join_lists(low_mask_list)

        assert len(low_len_list) == len(low_mask_list) == len(low_k_list)

        low_mask_list = [x_m.unsqueeze(1) for x_m in low_mask_list]
        low_att_list = [self.only_attention(trg_q, low_k, mask=mask) for low_k, mask in zip(low_k_list, low_mask_list)]

        cated_low_att = torch.cat(low_att_list, dim=-1)
        cated_low_v = torch.cat(low_v_list, dim=-2)

        sub_guider_att_list = torch.split(guider_att, 1, dim=-1)
        sub_guider_att_list = [sh.expand((-1, -1, -1, low_len_list[i])) for i, sh in enumerate(sub_guider_att_list)]

        guider_att_expand = torch.cat(sub_guider_att_list, dim=-1)

        final_att = guider_att_expand * cated_low_att
        output = torch.matmul(final_att, cated_low_v)

        output = output.transpose(1, 2).contiguous().view(batch_size, trg_len, -1)
        output = self.dropout(self.fc(output))
        output += residual
        return output, final_att


class DecDiffK(nn.Module):
    def __init__(self, n_head, d_model, d_k, n_modals):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.n_modals = n_modals
        self.guider_w_ks_list = nn.ModuleList([nn.Linear(d_model, n_head * d_k, bias=False)
                                            for _ in range(n_modals)])

    def forward(self, guider_rep, modals_num_list):
        batch_size = guider_rep.size(0)
        guider_len = guider_rep.size(1)

        guider_rep_list = torch.split(guider_rep, split_size_or_sections=modals_num_list, dim=1)
        guider_k_list = []
        for i, w_ks in enumerate(self.guider_w_ks_list):
            guider_k_list.append(w_ks(guider_rep_list[i]))

        guider_k = torch.cat(guider_k_list, dim=1)
        guider_k = guider_k.view(batch_size, guider_len, self.n_head, self.d_k)
        return guider_k

