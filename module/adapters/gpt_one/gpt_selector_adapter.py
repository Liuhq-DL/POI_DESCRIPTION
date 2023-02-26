import torch
from torch import nn
import math
from module.decoder_sub_modules import SelectorModule

from transformers.models.gpt2.modeling_gpt2 import (
Conv1D,

)


class SelectorAdapter():

    def init_adapter_module(self, ):
        print("initiate selector decoder adapter ")
        self.adapters = nn.ModuleDict()

    def add_adapter(self, layer_adapter_config, adapter_init_range):
        self.adapters["en_de_att"] = SelectorModule(**layer_adapter_config)
        init_model(self.adapters["en_de_att"], adapter_init_range)

    def adapter_layer_forward(self, hidden_states, mask, adapter_inputs):
        if len(self.adapters) == 0:
            return hidden_states
        guider_rep, guider_mask, low_rep_list, low_mask_list = adapter_inputs
        output_states, *_ = self.adapters["en_de_att"](hidden_states, guider_rep, guider_mask, low_rep_list, low_mask_list)
        return output_states


def init_model(model, init_range):
    for module in model.modules():
        init_weights(module, init_range)


def init_weights(module, init_range):
    if isinstance(module, (nn.Linear, Conv1D)):
        module.weight.data.normal_(mean=0.0, std=init_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=init_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


