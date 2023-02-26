import torch
from torch import nn
import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
from config.CONSTANTS import *
from transformers import BertTokenizer, BertModel


class BertEncoder1(nn.Module):
    def __init__(self, pre_path, bert_model, d_model):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(pre_path)
        self.linear = nn.Linear(bert_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, bert_inputs):
        outputs = self.bert_model(**bert_inputs)
        reps = outputs["last_hidden_state"]
        reps = self.linear(reps)
        reps = self.layer_norm(reps)
        return reps


