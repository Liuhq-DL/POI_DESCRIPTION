import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
import torch
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 split,
                 data_path,
                 pois_value_type
                 ):

        used_data_path = data_path.format(split)
        print(used_data_path)
        with open(used_data_path, "r", encoding="utf-8") as f:
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


