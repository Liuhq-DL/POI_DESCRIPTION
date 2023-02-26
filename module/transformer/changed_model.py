import torch
import torch.nn as nn
import numpy as np
from module.transformer.Layers import EncoderLayer, DecoderLayer
from config.CONSTANTS import *
from module.transformer.Models import PositionalEncoding
from utils.utils_b import pad_from_num_list, lng_lat_to_meter
import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../../"))
sys.path.append(dir_name)
from config.CONSTANTS import *


class NearTransEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.d_word_vec = d_word_vec

    def forward(self, poi_type, loc_emb, near_pois_num_list, type_token_num, return_attns=False):

        enc_slf_attn_list = []
        device = poi_type.device
        enc_output = self.src_word_emb(poi_type)
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output += loc_emb.unsqueeze(1)

        rep_tensor = torch.tensor([REPRESENT_INDEX], dtype=torch.long, device=device)
        rep_embedding = self.src_word_emb(rep_tensor)

        this_dtype = rep_embedding.dtype

        batched_enc_output_list = []
        pad_num = []
        c = 0
        max_batch_len = 0
        for n in near_pois_num_list:
            length = sum(type_token_num[c: c+n])
            c += n
            max_batch_len = max(length, max_batch_len)
        max_batch_len += 1
        curr = 0
        t_i = 0
        for n in near_pois_num_list:
            tmp_batch = enc_output[curr: curr+n, :, :]
            tmp_list = [rep_embedding]
            for x in tmp_batch:
                tmp_list.append(x[:type_token_num[t_i], :])
                t_i += 1
            tmp_list_tensor = torch.cat(tmp_list, dim=0)
            batch_lenght = len(tmp_list_tensor)
            pad_num.append(batch_lenght)
            if batch_lenght < max_batch_len:
                tmp_list_tensor = torch.cat([tmp_list_tensor,
                                             torch.zeros((max_batch_len-batch_lenght, self.d_word_vec),
                                                         device=device, dtype=this_dtype)])
            batched_enc_output_list.append(tmp_list_tensor)
            curr += n
        batched_enc_output = torch.stack(batched_enc_output_list, dim=0)

        assert t_i == len(type_token_num)

        final_mask = pad_from_num_list(pad_num)

        enc_output = self.layer_norm(batched_enc_output)
        final_mask = final_mask.to(device)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=final_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, final_mask, enc_slf_attn_list
        return enc_output, final_mask


class LearnIterpPosEncode2D(nn.Module):
    def __init__(self, side_num, vec_len, side_len):
        super().__init__()
        self.pos_emb = nn.Embedding(side_num**2, vec_len)
        self.side_len = side_len
        self.side_num = side_num

    def forward(self, locations):
        device = locations.device
        eposion1 = 1e-8
        locations += self.side_len / 2
        locations = torch.clamp(locations, eposion1, self.side_len - eposion1)
        locations = torch.round(locations * (self.side_num-1) / self.side_len)
        locations = locations.type(torch.LongTensor)
        locations = locations.to(device)
        location_ids = (locations[:, 0] * self.side_num) + locations[:, 1]
        location_emb = self.pos_emb(location_ids)
        return location_emb


class MLPLayer(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_inner)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        res = x
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        x += res
        x = self.layer_norm(x)
        return x

