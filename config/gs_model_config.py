import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
from config.CONSTANTS import *
from data.pre_func import Tokenlize
import torch
config_name = "gs_model_config.py"


dataset_config = {
    "data_path": os.path.join(DATA_ROOT_PATH, "part_{}_samples.json"),
    "pois_value_type": ["type", "location"],
}


token_config = {
    "src_vocab_path": os.path.join(DATA_ROOT_PATH, "vocabulary_src.json"),
    "trg_vocab_path": os.path.join(DATA_ROOT_PATH, "vocabulary_trg.json"),
    "tokenizer": {"name": "SynTokenlize",
                  "SynTokenlize": {}
                  },
    "pre_tokenizer": {
        "bert_tokenizer": {"name": "AutoTokenizer", "path": os.path.join(PRETRAINED_ROOT, "bert_base_chinese")},
        "decoder_tokenizer": {"name": "BertTokenizer",
                              "path": os.path.join(PRETRAINED_ROOT, "gpt2_distil_chinese_cluecorpussmall")}
    }
}

dataloader_config = {
    "train": {
        "batch_size": 16,
        "num_workers": 8},

    "dev": {
        "batch_size": 16,
        "num_workers": 8},

    "test": {
        "batch_size": 8,
        "num_workers": 4},

}

model_config = {
    "name": "GSModel",
    "use_pretrained": True,
    "decoder_pretrained": True,
    "encoder_config":{
        "bert_train_layer_ids": [9, 10, 11],
        "one_fea_near": True,
        "low_bert_encoder_config":{
         "pre_path": os.path.join(PRETRAINED_ROOT, "bert_base_chinese"),
         "bert_model": 768,
         "d_model": 768
        },
        "guider_config":{
         "n_layers": 2,
         "n_head": 12,
         "d_k": 64,
         "d_v": 64,
         "d_model": 768,
         "d_inner": 3072,
         "n_modals": 3,
         "dropout": 0.1,
         "scale_emb": False
        },
        "near_trans_encoder_config":{
         "d_word_vec": 768,
         "n_layers": 6,
         "n_head": 12,
         "d_k": 64,
         "d_v": 64,
         "d_model": 768,
         "d_inner": 3072,
         "dropout": 0.1,
         "n_position": 200,
        },
        "center_type_bert_encoder_config": {
         "pre_path": os.path.join(PRETRAINED_ROOT, "bert_base_chinese"),
         "bert_model": 768,
         "d_model": 768
        },
        "loc_pos_encode_config": {
            "side_num": 25,
            "vec_len": 768,
            "side_len": SEARCH_RADIOUS * 2
        }
    },
     "decoder_config": {
         "pre_path": os.path.join(PRETRAINED_ROOT, "gpt2_distil_chinese_cluecorpussmall"),
         "layer_adapter_config": {
             "n_head": 12,
             "d_model": 768,
             "d_k": 64,
             "d_v": 64,
             "dropout": 0.1
         },
         "add_layer_ids": [1, 3, 5],
         "adapter_init_range": 0.02,
         "adapter_name": "SelectorAdapter",
         "vocab_size": 21128,
         "pad_id": 0,
         "unk_id": 100,
         "sep_id": 102
    },
    "sub_module_config":{
        "modals_num": 3
    },
    "loss_config": {
        "name": "CrossEntropyLoss",
        "CrossEntropyLoss": {
            "reduction": "sum",
            "ignore_index": 0,
            "label_smoothing": 0.1
        }
    },
    "data_processor": {
        "name": "GSProcess",
        "GSProcess": {
            "review_max_len": 350,
            }
    },
}


generate_config = {
            "beam_size": 8,
            "decode_min_len": 60,
            "decode_max_len": 260,
            "force_eos": 120,
            "avoid_ngram": 2,
            "to_word": True
}


train_config = {
    "init_lr": 0.001,
    "stage_lr_list": [0.001, 0.0005],
    "optimizer": "AdamW",
    "AdamW": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False
    },
    "Adam": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False
    },
    "clip_norm": 8.0,
    "clip_value": -5.0,
    "out_dir": os.path.join(DATA_ROOT_PATH, "../out_dir/sg_save_dir"),
    "schedule_config": {
        "decay_rate": 0.9,
        "patience": 1
    },
    "early_stop_patience": 10,
    "log_iters": 20,
    "resume": -1,
    "save_optim": True,
    "max_epoch": 400,
    "valid_iters": 150,
    "save_min_iter": 100,
    "max_save_num": 5,
    "use_cuda": True,
    "gpu_ids": "0",
    "seed_value": 1,
    "update_interval": 3,
    "distributed": False,
    "dist_backend": "nccl",
    "parallel": False

}

test_config={
    "generate_save_path": os.path.join(train_config["out_dir"], "generated_text.json"),
    "model_name": BEST_MODEL_MANE,
    "use_cuda": True
}




