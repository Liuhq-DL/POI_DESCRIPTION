import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from config.CONSTANTS import *
from module.transformer.changed_model import NearTransEncoder, LearnIterpPosEncode2D
from config import model_config, generate_config
from utils.beam_search_pretrained import pretrained_beam_search
from utils.utils_b import text_id2_tensor, pad_from_num_list, location_to_tensor, get_location_offset, get_text_sum,\
    get_location_meters, cated_to_batched, change_parms_grad
from config import dataset_config
from module.diff_encoder import DiffGuiderEncoder
from module.bert_modules import BertEncoder1
from module.adapters.gpt_one.gpt2_adapter_module import AdapterGPT2Decoder
import json
from data.process_func import process_list


class GSModel(nn.Module):
    def __init__(self, src_vocab, trg_vocab, pre_tokenizer_dict):
        super().__init__()
        self.encoder = GuiderFuseEncoder(src_vocab)
        self.decoder = AdapterGPT2Decoder(
            **model_config["decoder_config"]
        )
        self.decoder_d_model = model_config["decoder_config"]["layer_adapter_config"]["d_model"]
        self.trg_vocab = trg_vocab
        self.decoder_vocab_size = model_config["decoder_config"]["vocab_size"]

        self.decoder_tokenizer = pre_tokenizer_dict["decoder_tokenizer"]

        self.dec_pad_id = model_config["decoder_config"]["pad_id"]
        self.dec_unk_id = model_config["decoder_config"]["unk_id"]
        self.dec_sep_id = model_config["decoder_config"]["sep_id"]

    def forward(self, review_inputs, num_list, decoder_inputs,
                cated_type_tensor, cated_loc_tensor, near_pois_num_list, type_token_num,
                center_type_inputs
                ):
        guider_output, guider_mask,\
        low_rep_list, low_mask_list = self.encoder(review_inputs, num_list,
                                                       cated_type_tensor, cated_loc_tensor, near_pois_num_list,
                                                       type_token_num,
                                                       center_type_inputs
                                                       )

        adapter_inputs = (guider_output, guider_mask, low_rep_list, low_mask_list)

        logist = self.decoder(decoder_inputs, adapter_inputs)
        logist = logist[:, :-1, :]
        return logist

    def generate(self, review_inputs, num_list,
                        cated_type_tensor, cated_loc_tensor, near_pois_num_list, type_token_num,
                        center_type_inputs
                 ):
        total_encoded_output, guider_mask, low_rep_list, low_mask_list = self.encoder(review_inputs, num_list,
                                                       cated_type_tensor, cated_loc_tensor, near_pois_num_list,
                                                       type_token_num,
                                                       center_type_inputs
                                                       )
        context_tuple = (total_encoded_output, guider_mask, low_rep_list, low_mask_list)
        generated_text, sort_score = pretrained_beam_search(model=self, context_tuple=context_tuple,
                                                            pad_index=self.dec_pad_id,
                                                            unk_index=self.dec_unk_id,
                                                            end_index=self.dec_sep_id,
                                                            **generate_config)
        return generated_text, sort_score

    def single_decode(self, context_tuple, trg_input_seq):
        encoded_vec, guider_mask, low_rep_list, low_mask_list = context_tuple
        adapter_inputs = (encoded_vec, guider_mask, low_rep_list, low_mask_list)
        device = encoded_vec.device

        trg_string = self.decoder_tokenizer.batch_decode(trg_input_seq, skip_special_tokens=True)
        decoder_inputs = self.decoder_tokenizer(trg_string, return_tensors='pt', padding=True)
        for i, v in decoder_inputs.items():
            decoder_inputs[i] = v.to(device)

        logist = self.decoder(decoder_inputs, adapter_inputs)
        logist = logist[:, -2, :]
        logist = F.log_softmax(logist, dim=-1)
        return logist


class GuiderFuseEncoder(nn.Module):
    def __init__(self, src_vocab):
        super().__init__()
        self.review_encoder = BertEncoder1(
            **model_config["encoder_config"]["low_bert_encoder_config"]
        )

        self.guider = DiffGuiderEncoder(
            **model_config["encoder_config"]["guider_config"]
        )

        self.near_pois_encoder = NearTransEncoder(
            n_src_vocab=len(src_vocab),
            pad_idx=PAD_INDEX,
            **model_config["encoder_config"]["near_trans_encoder_config"]
        )

        self.center_type_encoder = BertEncoder1(
            **model_config["encoder_config"]["center_type_bert_encoder_config"]
        )

        self.loc_pos_emb = LearnIterpPosEncode2D(
            **model_config["encoder_config"]["loc_pos_encode_config"]
        )

        self.one_fea_near = model_config["encoder_config"]["one_fea_near"]

    def forward(self, review_inputs, review_num_list,
                cated_type_tensor,  cated_loc_tensor, near_pois_num_list, type_token_num,
                center_type_inputs):
        device = review_inputs["input_ids"].device

        cated_reivews_encoded = self.review_encoder(review_inputs)

        max_len = max(review_num_list)
        reviews_mask = review_inputs["attention_mask"]
        reviews_mask = reviews_mask.unsqueeze(-2).type(torch.bool)

        batched_review_list = cated_to_batched(cated_reivews_encoded, review_num_list)
        batched_mask_list = cated_to_batched(reviews_mask, review_num_list)

        rep_tensor = cated_reivews_encoded[:, 0, :]
        vec_len = rep_tensor.shape[-1]

        review_tensors_list = []
        curr = 0
        for n in review_num_list:
            tmp_tensor = rep_tensor[curr: curr + n, :]
            if len(tmp_tensor) < max_len:
                pad_tensor = torch.zeros((max_len - len(tmp_tensor), vec_len), dtype=rep_tensor.dtype,
                                         device=device)
                tmp_tensor = torch.cat([tmp_tensor, pad_tensor], dim=0)
            review_tensors_list.append(tmp_tensor)
            curr += n

        assert curr == sum(review_num_list)

        loc_emb = self.loc_pos_emb(cated_loc_tensor)

        near_pois_encoded, near_final_mask = self.near_pois_encoder(cated_type_tensor, loc_emb,
                                                   near_pois_num_list, type_token_num)
        near_pois_rep = near_pois_encoded[:, 0, :].unsqueeze(1)

        center_pad_mask = center_type_inputs["attention_mask"]
        center_pad_mask = center_pad_mask.unsqueeze(-2).type(torch.bool)
        center_type_encoded = self.center_type_encoder(center_type_inputs)
        center_type_rep = center_type_encoded[:, 0, :].unsqueeze(1)

        review_rep_tensor = torch.stack(review_tensors_list, dim=0) # (batch_size, max len, rep_vec_len)
        rep_list = [near_pois_rep, center_type_rep,  review_rep_tensor]

        modal_len_list = [t_x.size(1) for t_x in rep_list]

        guider_tensor = torch.cat(rep_list, dim=1)
        for i in range(len(review_num_list)):
            review_num_list[i] += 2
        guider_mask = pad_from_num_list(num_list=review_num_list)
        guider_mask = guider_mask.to(device)

        guider_output, *_ = self.guider(guider_tensor, modal_len_list, guider_mask)  # (batch_size, max review num, rep vec len)

        if self.one_fea_near:
            near_pois_encoded = near_pois_encoded[:, 0, :].unsqueeze(1)
            near_final_mask = near_final_mask[:, :, 0].unsqueeze(-1)
        low_rep_list = [[near_pois_encoded], [center_type_encoded], batched_review_list]
        low_mask_list = [[near_final_mask], [center_pad_mask], batched_mask_list]

        return guider_output, guider_mask, low_rep_list, low_mask_list


class GSProcess():
    def __init__(self, src_vocab, trg_vocab, tokenizer, device, pre_tokenizer_dict, review_max_len,
                 ):
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.review_max_len = review_max_len
        self.device = device

        self.bert_tokenizer = pre_tokenizer_dict["bert_tokenizer"]
        self.decoder_tokenizer = pre_tokenizer_dict["decoder_tokenizer"]

        self.epoch = -1
        self.iteration = -1

    def __call__(self, data):
        batch_reviews_list = data["review"]
        batch_target = data["reference"]

        batch_target = ["".join(self.tokenizer(t, True)) for t in batch_target]
        decoder_inputs = self.decoder_tokenizer(batch_target, return_tensors='pt', padding=True)
        for i, v in decoder_inputs.items():
            decoder_inputs[i] = v.to(self.device)
        out_batch_trg_tensor = decoder_inputs["input_ids"][:, 1:]

        refer_token_sum = get_text_sum(out_batch_trg_tensor.tolist())

        num_list = [len(reviews) for reviews in batch_reviews_list]
        cated_review_list = []
        for reviews in batch_reviews_list:
            cated_review_list += reviews

        cated_review_list = ["".join(self.tokenizer(r, True)) for r in cated_review_list]
        review_inputs = self.bert_tokenizer(cated_review_list, return_tensors='pt', padding=True)

        for i, v in review_inputs.items():
            review_inputs[i] = v.to(self.device)
        ###

        batch_near_pois = data["near_pois"]  # (batch_size, near num,  item num)
        cated_type_str = []
        cated_location = []
        near_pois_num_list = []

        center_lng = [float(x) for x in data["lng"]]
        center_lat = [float(x) for x in data["lat"]]

        poi_value_type = dataset_config["pois_value_type"]
        type_str_index = poi_value_type.index("type")
        location_index = poi_value_type.index("location")
        for i, bd in enumerate(batch_near_pois):
            near_pois_num_list.append(len(bd))
            for item in bd:
                cated_location.append(get_location_meters(item[location_index],
                                                          center_lng[i],
                                                          center_lat[i]))
                cated_type_str.append(item[type_str_index])

        cated_type_id = [[self.src_vocab(t) for t in self.tokenizer(s_t, False)] for s_t in cated_type_str]
        type_token_num = [len(tl) for tl in cated_type_id]
        cated_type_tensor = text_id2_tensor(cated_type_id, len(cated_type_id))

        cated_loc_tensor = torch.stack(cated_location, dim=0)

        cated_loc_tensor = cated_loc_tensor.to(self.device)
        cated_type_tensor = cated_type_tensor.to(self.device)

        center_type = data["category"]
        center_type_inputs = self.bert_tokenizer(center_type, return_tensors='pt', padding=True)
        for i, v in center_type_inputs.items():
            center_type_inputs[i] = v.to(self.device)

        return review_inputs, num_list, decoder_inputs, out_batch_trg_tensor, refer_token_sum,\
               cated_type_tensor,  cated_loc_tensor, near_pois_num_list, type_token_num, center_type_inputs

    @staticmethod
    def data_to_forward(data_list):
        return data_list[0:3] + data_list[5:10]

    @staticmethod
    def data_output_to_criterion(data_list, model_output):
        trg_tensor = data_list[3]
        logist = model_output
        logist = logist.contiguous().view(-1, logist.size(-1))
        trg_tensor = trg_tensor.contiguous().view(-1)
        return logist, trg_tensor

    @staticmethod
    def data_to_generate(data_list):
        return data_list[0: 2] + data_list[5:10]

    @staticmethod
    def data_to_token_sum(data_list):
        return data_list[4]

    @staticmethod
    def set_model_stage(model, stage):
        if stage == 0:
            change_parms_grad(model.encoder.review_encoder.bert_model, False)
            change_parms_grad(model.encoder.center_type_encoder.bert_model, False)

            change_parms_grad(model.decoder, False)
            for i, layer in model.decoder.iter_layers():
                change_parms_grad(layer.adapters, True)
                if len(layer.adapters) > 0:
                    change_parms_grad(layer.ln_2, True)

        elif stage == 1:
            bert_train_layer_ids = model_config["encoder_config"]["bert_train_layer_ids"]
            for i, bert_layer in enumerate(model.encoder.review_encoder.bert_model.encoder.layer):
                if i in bert_train_layer_ids:
                    change_parms_grad(bert_layer, True)

            for i, bert_layer in enumerate(model.encoder.center_type_encoder.bert_model.encoder.layer):
                if i in bert_train_layer_ids:
                    change_parms_grad(bert_layer, True)
        else:
            raise RuntimeError

    def switch_stage(self, model, epoch, iteration):
        changed = False
        stage = None
        if epoch == 0 and self.epoch == -1:
            stage = 0
            self.set_model_stage(model, stage=stage)
            print("train in status: 0")
            changed = True

        if epoch == 2 and self.epoch == 1:
            stage = 1
            self.set_model_stage(model, stage=stage)
            print("train in status 1")
            changed = True

        self.epoch = epoch
        self.iteration = iteration
        return changed, stage



