import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
from config.CONSTANTS import *
import numpy as np
from utils.train_utils import Terminater, MetricLogger, get_logger, Schedule1, remove_all, config_backup, set_seed
from config import train_config, dataset_config, config_name, token_config, model_config
from data.vocab import Vocabulary
from data_torch.build_dataloader import get_loader
from train.trainer import Trainer
import time
import torch
from model import *
from train.test import test
from utils.utils_b import build_pre_tokenizer


def train():
    out_dir = train_config["out_dir"]
    if not os.path.isdir(out_dir):
        os.system("mkdir -p %s" % out_dir)
    else:
        print("exist save path: ", out_dir)
        if not USE_NOHUP:
            x = input("if continue, y/n")
            if x != "y":
                exit()

    logger = get_logger(os.path.join(out_dir, "logger.txt"))
    logger.info(out_dir)
    logger.info("use config file: %s" % config_name)
    config_file_name = os.path.join(os.path.dirname(__file__), "../config", config_name)
    config_backup(config_file_name, out_dir)

    if train_config["parallel"]:
        logger.info("train use data parallel")

    src_vocab = Vocabulary(vocab_path=token_config["src_vocab_path"])
    trg_vocab = Vocabulary(vocab_path=token_config["trg_vocab_path"])

    loss_name = model_config["loss_config"]["name"]
    logger.info("use loss name: %s" % loss_name)
    criterion = eval(loss_name)(**model_config["loss_config"][loss_name])

    train_dataloader = get_loader(split="train")
    valid_dataloader = get_loader(split="dev")

    if torch.cuda.is_available() and train_config["use_cuda"]:
        logger.info("train use GPU")
        device = torch.device("cuda")
    else:
        logger.info("train use CPU")
        device = torch.device("cpu")

    tokenizer_name = token_config["tokenizer"]["name"]
    logger.info("use tokenizer: %s " % tokenizer_name)
    tokenizer = eval(tokenizer_name)(**token_config["tokenizer"][tokenizer_name])

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
    if decoder_pretrained:
        logger.info("use decoder pretrained model")

    logger.info("use model name: %s" % model_config["name"])
    model_parms_list = [src_vocab, trg_vocab]
    if decoder_pretrained:
        assert use_pretrained
        model_parms_list.append(pre_tokenizer_dict)
    model = eval(model_config["name"])(*model_parms_list)

    data_processor = eval(data_processor_name)(src_vocab=src_vocab,
                                               trg_vocab=trg_vocab,
                                               tokenizer=tokenizer,
                                               device=device,
                                               **processor_parm_dict
                                               )

    trainer = Trainer(model, criterion,
                       device=device,
                       data_processor = data_processor,
                       resume=train_config["resume"],
                       logger=logger,
                       parallel=train_config["parallel"])

    log_iters = train_config["log_iters"]
    valid_iters = train_config["valid_iters"]
    update_iterval = train_config["update_interval"]
    save_min_iter = train_config["save_min_iter"]
    log_iters *= update_iterval
    valid_iters *= update_iterval
    save_min_iter *= update_iterval

    terminater = Terminater(out_dir, last_checkpoint=train_config["resume"],
                            checkpoints_patience=train_config["early_stop_patience"])
    metric_logger = MetricLogger(metrics_name=["loss", "perplexity"],
                                 save_path=os.path.join(train_config["out_dir"], "metrics.log"),
                                last_checkpoints=train_config["resume"], check_steps=log_iters)
    schedule = Schedule1(trainer.optimizer, logger=logger, **train_config["schedule_config"])
    token_sum = 0
    loss_sum = 0

    total_train_step = len(train_dataloader)
    iteration = train_config["resume"]
    save_iter_list = []
    pre_epoch = train_config["resume"] // total_train_step

    if train_config["resume"] > 0:
        logger.info("resume to train, pre epoch: %s" % pre_epoch)
        terminater.load()
        metric_logger.load()

    best_val_ppl = 10**10

    begin_time = time.time()
    for epoch in range(train_config["max_epoch"]):
        if epoch <= pre_epoch:
            continue
        for i, data in enumerate(train_dataloader):
            iteration += 1
            if use_pretrained:
                if_changed = trainer.get_train_info(epoch, iteration)
                schedule.reset(trainer.optimizer, if_changed)
            train_loss, refer_token_num, grad_norm = trainer.train_one_batch(data)
            train_ppl = np.exp(train_loss/refer_token_num).item()
            token_sum += refer_token_num
            loss_sum += train_loss

            if iteration % log_iters == 0:
                logger.info('epoch %d, iter %d, avg. loss %.6f, avg. ppl %.2f ' \
                      ', time elapsed %.2f seconds, total inters: %d'
                      % (epoch, iteration,
                         train_loss/refer_token_num,
                         train_ppl,
                         time.time() - begin_time, total_train_step))
                metric_logger.step(metrics={"loss": train_loss/refer_token_num, "perplexity": train_ppl})
                loss_sum = 0
                token_sum = 0

            if iteration%valid_iters == 0 and iteration > 0:
                logger.info("reach checkpoint, start to validation")
                val_begin = time.time()
                val_avg_loss, val_avg_ppl = valid(trainer, valid_dataloader)
                logger.info('Validation: epoch %d, iter %d, dev. loss %.2f, dev. ppl %.2f, time consusme: %.2f' % (
                epoch, iteration, val_avg_loss, val_avg_ppl, time.time()-val_begin))

                if val_avg_ppl < best_val_ppl:
                    best_val_ppl = val_avg_ppl

                schedule.step(val_avg_loss)
                not_improve, not_improve_than_last, if_terminate = terminater.step(val_avg_ppl)

                if iteration > save_min_iter:
                    if len(save_iter_list) >= train_config["max_save_num"]:
                        logger.info("delete old checkpoint:  %d"%save_iter_list[0])
                        trainer.del_check_point(save_iter_list[0], save_optim=train_config["save_optim"])
                        save_iter_list = save_iter_list[1:]
                    logger.info("save checkpoint: %d"%iteration)
                    trainer.save_checkpoint(iteration, save_optim=train_config["save_optim"])
                    save_iter_list.append(iteration)
                    terminater.save()
                    metric_logger.save()

                    if not not_improve:
                        logger.info("save current best model")
                        save_best_checkpoint(out_dir, iteration)

                if if_terminate:
                    logger.info("hit patience: %d, stop training " % terminater.current_no_improve)
                    logger.info("best val ppl: %.2f" % best_val_ppl)
                    remove_all(MODEL_NAME, OPTIMIZER_NAME, out_dir)
                    return

    logger.info("reach max epoch, stop training")
    logger.info("best val ppl: %.2f" % best_val_ppl)
    remove_all(MODEL_NAME, OPTIMIZER_NAME, out_dir)


def save_best_checkpoint(out_dir, iteration):
    model_path = os.path.join(out_dir, MODEL_NAME.format(iteration))
    best_path = os.path.join(out_dir, BEST_MODEL_MANE)
    os.system("cp %s %s"%(model_path, best_path))


def valid(trainer, valid_dataloader):
    loss_total = 0
    tokens_num_total = 0
    for data in valid_dataloader:
        val_loss, trg_len = trainer.valid_one_batch(data)
        loss_total += val_loss
        tokens_num_total += trg_len

    return loss_total/tokens_num_total, np.exp(loss_total/tokens_num_total)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = train_config["gpu_ids"]
    set_seed(train_config["seed_value"])
    train()
    test()


