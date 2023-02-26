import os
import random
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
from config.CONSTANTS import *
import torch
import numpy as np
import math
from transformers import AutoTokenizer, BertTokenizer


def com_acc(output, label):
    output = output.detach().cpu().numpy()
    predict = np.argmax(output, axis=1)
    label = label.detach().cpu().numpy()
    acc = np.sum(predict == label) / len(label)
    return acc


def text_id2_tensor(text_id_list, batch_size):
    length_list = [len(rb) for rb in text_id_list]
    body_tensor = torch.full((batch_size, max(length_list)), PAD_INDEX)
    for i, b in enumerate(text_id_list):
        body_tensor[i, 0: length_list[i]] = torch.tensor(b)
    body_tensor.long()
    return body_tensor


def pad_from_num_list(num_list):
    max_length = max(num_list)
    batch_size = len(num_list)
    mask = torch.full((batch_size, max_length), fill_value=False)

    for i, nl in enumerate(num_list):
        mask[i, 0:nl] = torch.full((nl, ), fill_value=True)
    mask = mask.unsqueeze(-2)
    return mask


def location_to_tensor(location):
    lng, lat = location.strip().split(",")
    lng = float(lng.strip())
    lat = float(lat.strip())
    return torch.tensor([lng, lat])


def get_location_offset(location_str, center_lng, center_lat):
    location_tensor = location_to_tensor(location_str)
    loc_offset = location_tensor - torch.tensor([center_lng, center_lat])
    return loc_offset


def get_text_sum(text_batch):
    total = 0
    for text in text_batch:
       total += len(text)
    return total


def lng_lat_to_meter(lng_diff, lat_diff, lat_center):
    R = 6371000
    lng_meter = lng_diff * math.cos(lat_center / 180 * math.pi) * math.pi * R / 180
    lat_meter = lat_diff * math.pi * R / 180

    return lng_meter, lat_meter


def get_location_meters(location_str, center_lng, center_lat):
    loc_offect = get_location_offset(location_str, center_lng, center_lat).tolist()
    meters = lng_lat_to_meter(loc_offect[0], loc_offect[1], center_lat)
    return torch.tensor(meters)


def join_lists(seq_list):
    result = []
    for x in seq_list:
        result += x
    return result


def cated_to_batched(cated_tensor, num_list):
    device = cated_tensor.device
    max_num = max(num_list)
    seq_len = cated_tensor.size(-2)
    seq_vec_len = cated_tensor.size(-1)
    splited_tensor = list(torch.split(cated_tensor, num_list, dim=0))

    for i, n in enumerate(num_list):
        st = splited_tensor[i]
        if len(st) < max_num:
            pad_tensor = torch.zeros((max_num - len(st), seq_len, seq_vec_len),
                                     dtype=cated_tensor.dtype,
                                     device=device)
            st = torch.cat([st, pad_tensor], dim=0)
            splited_tensor[i] = st

    batched_total = torch.stack(splited_tensor, dim=0)
    batched_list = torch.split(batched_total, 1, dim=1)
    batched_list = [x.squeeze(1) for x in batched_list]
    return batched_list


def split_list_by_nums(given_list, nums_list):
    assert sum(nums_list) == len(given_list)
    result = []
    i = 0
    k = 0
    tmp = []
    for x in given_list:
        i += 1
        tmp.append(x)
        if i == nums_list[k]:
            result.append(tmp)
            tmp = []
            k += 1
            i = 0

    assert k == len(nums_list) == len(result)
    return result


def extant_mask(mask, extant_num, extant_value=True):
    device = mask.device
    batch_size = mask.size(0)
    att_mask = torch.full((batch_size, 1, extant_num), fill_value=extant_value, device=device)
    result_mask = torch.cat([att_mask, mask], dim=-1)
    return result_mask


def multiple_nums(num_list, multiple):
    new_num_list = []
    for n in num_list:
        new_num_list += [n] * multiple

    return new_num_list


def single_pre_tokenizer(single_dict):
    model_name = single_dict["name"]
    save_path = single_dict["path"]
    tokenizer = eval(model_name).from_pretrained(save_path)
    return tokenizer


def build_pre_tokenizer(tokenizer_dict):
    new_dict = {}
    for k, v in tokenizer_dict.items():
        new_dict[k] = single_pre_tokenizer(v)
    return new_dict


def change_parms_grad(model, require_grad):
    for n, p in model.named_parameters():
        p.requires_grad = require_grad

