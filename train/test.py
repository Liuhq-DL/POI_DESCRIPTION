import json
import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
import torch
from torch import nn
import torch.nn.functional as F
from config.CONSTANTS import *
from data_torch.build_dataloader import get_loader
from data.vocab import Vocabulary
from config import train_config, model_config, dataset_config, test_config, token_config
import time
from utils.train_utils import get_logger
from model import *
from utils.utils_b import build_pre_tokenizer


def test():
    logger = get_logger(os.path.join(train_config["out_dir"], "logger_test.txt"))
    print("start to generate")
    logger.info("out dir: %s" % train_config["out_dir"])
    start_time = time.time()
    test_loader = get_loader(split="test")
    src_vocab = Vocabulary(vocab_path=token_config["src_vocab_path"])
    trg_vocab = Vocabulary(vocab_path=token_config["trg_vocab_path"])

    data_processor_name = model_config["data_processor"]["name"]
    logger.info("use data processor name: %s" % data_processor_name)

    processor_parm_dict = model_config["data_processor"][data_processor_name].copy()
    use_pretrained = model_config["use_pretrained"] if "use_pretrained" in model_config else False
    if use_pretrained:
        logger.info("use pretrained model")
        pre_tokenizer_dict = build_pre_tokenizer(token_config["pre_tokenizer"])
        print(pre_tokenizer_dict)
        processor_parm_dict["pre_tokenizer_dict"] = pre_tokenizer_dict

    decoder_pretrained = model_config["decoder_pretrained"] if "decoder_pretrained" in model_config else False

    logger.info("use model name: %s" % model_config["name"])
    model_parms_list = [src_vocab, trg_vocab]
    if decoder_pretrained:
        assert use_pretrained
        model_parms_list.append(pre_tokenizer_dict)
    model = eval(model_config["name"])(*model_parms_list)

    map_location = None if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(os.path.join(train_config["out_dir"], test_config["model_name"]),
                                     map_location=map_location))

    device = torch.device("cuda:0" if torch.cuda.is_available() and test_config["use_cuda"] else "cpu")
    model.to(device)
    model.eval()

    tokenizer_name = token_config["tokenizer"]["name"]
    logger.info("use tokenizer: %s " % tokenizer_name)
    tokenizer = eval(tokenizer_name)(**token_config["tokenizer"][tokenizer_name])

    data_processor = eval(data_processor_name)(src_vocab=src_vocab,
                                               trg_vocab=trg_vocab,
                                               tokenizer=tokenizer,
                                               device=device,
                                               **processor_parm_dict
                                               )

    result_dict = {}
    for data in test_loader:
        poi_id_list = data["poi_id"]
        data_processed = data_processor(data)
        generate_input = data_processor.data_to_generate(data_processed)

        with torch.no_grad():
            batch_beam_text, top_score = model.generate(*generate_input)

        if decoder_pretrained:
            generated_text = ["".join(beam[0]) for beam in batch_beam_text]
        else:
            generated_text = ["".join(beam[0][1:]) for beam in batch_beam_text]

        for k, g in zip(poi_id_list, generated_text):
            result_dict[k] = g

    with open(test_config["generate_save_path"], "w", encoding=ENCODING) as f:
        json.dump(result_dict, f)
    print("time consume: ", (time.time()-start_time) / 60)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = train_config["gpu_ids"]
    test()


