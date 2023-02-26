import os
import sys
import torch
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
from torch.utils.data import DataLoader
from config import dataset_config, dataloader_config
from data_torch.dataset import Dataset


def get_loader(split="train", distributed=False, num_gpus=1, rank=0):

    dataset = Dataset(
        split=split,
        **dataset_config
    )
    if_shuffle = True if split == "train" else False
    if not distributed:
        dataloader = DataLoader(dataset,
                                shuffle=if_shuffle,
                                collate_fn=collate_func,
                                **dataloader_config[split])
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=num_gpus,
            rank=rank,
            shuffle=if_shuffle
        )
        dataloader = DataLoader(dataset,
                                shuffle=False,
                                collate_fn=collate_func,
                                sampler=dist_sampler,
                                **dataloader_config[split])

    return dataloader


def collate_func(data_dict_list):
    data_batch_dict = {}
    for k in data_dict_list[0].keys():
        data_batch_dict[k] = []
    for dd in data_dict_list:
        for k, v in dd.items():
            data_batch_dict[k].append(v)
    return data_batch_dict

