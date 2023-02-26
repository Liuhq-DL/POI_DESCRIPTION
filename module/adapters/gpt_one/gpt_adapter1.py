import torch
from torch import nn
from config import model_config
import math

from module.adapters.gpt_one.gpt_selector_adapter import SelectorAdapter


class AdapterBase():
    def __init__(self):
        super().__init__()
        self.adapter_name = model_config["decoder_config"]["adapter_name"]
        print("use adapter name: ", self.adapter_name)
        self.adapter_class = eval(self.adapter_name)

    def init_adapter_module(self, *args, **kwargs):
        self.adapter_class.init_adapter_module(self, *args, **kwargs)

    def add_adapter(self, *args, **kwargs):
        self.adapter_class.add_adapter(self, *args, **kwargs)

    def adapter_layer_forward(self, *args, **kwargs):
        return self.adapter_class.adapter_layer_forward(self, *args, **kwargs)

