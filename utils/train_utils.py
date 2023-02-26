import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import torch
from config.CONSTANTS import *
import random


class Terminater():
    def __init__(self, model_path, last_checkpoint,  checkpoints_patience, min_mode=True):
        self.checkpoints_patience = checkpoints_patience
        self.min_mode = min_mode
        self.current_no_improve = 0
        self.best_metric = None
        self.not_improve = False
        self.path = os.path.join(model_path, "terminater_vars.pt")
        if last_checkpoint > 0:
            self.load()
        self.pre_metric = None

    def improved(self):
        return not self.not_improve

    def save(self):
        torch.save(vars(self), self.path)

    def load(self):
        term_value = torch.load(self.path)
        self.checkpoints_patience = term_value["checkpoints_patience"]
        self.min_mode = term_value["min_mode"]
        self.current_no_improve = term_value["current_no_improve"]
        self.best_metric = term_value["best_metric"]
        self.not_improve = term_value["not_improve"]
        self.path = term_value["path"]
        self.pre_metric = term_value["pre_metric"]

    def step(self, metric):

        if not self.best_metric or not self.pre_metric:
            self.best_metric = metric
            self.pre_metric = metric
            return False, False, False

        if self.min_mode:
            self.not_improve = metric >= self.best_metric
            self.best_metric = min(self.best_metric, metric)
            not_improve_than_last = metric >= self.pre_metric
        else:
            self.not_improve = metric <= self.best_metric
            self.best_metric = max(self.best_metric, metric)
            not_improve_than_last = metric <= self.pre_metric

        if self.not_improve:
            self.current_no_improve += 1
        else:
            self.current_no_improve = 0

        finish = False
        if self.current_no_improve > self.checkpoints_patience:
            finish = True
        self.pre_metric = metric

        return self.not_improve, not_improve_than_last, finish


class MetricLogger():
    def __init__(self, metrics_name: list, save_path: str, last_checkpoints = 0, check_steps = 0 ):
        self.metrics = {}
        for metric in metrics_name:
            self.metrics[metric] = []

        self.metrics["check_steps"] = check_steps
        self.save_path = save_path
        self.pre_steps = last_checkpoints//check_steps
        if last_checkpoints > 0:
            self.load()

    def save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.metrics, f, 0)

    def load(self):
        with open(self.save_path, "rb") as f:
            metrics = pickle.load(f)
        for name, value in metrics.items():
            if name == "check_steps":
                continue
            self.metrics[name] = value[:self.pre_steps]

    def step(self, metrics:{}):
        for name, value in metrics.items():
            self.metrics[name].append(value)

    def plot(self, metrics_name: list = None):
        if not metrics_name:
            metrics_name = list(self.metrics.keys())

        for i, metric in enumerate(metrics_name):
            assert metric in self.metrics.keys(), "metric not be logged"
            if metric == "check_steps":
                continue
            plt.figure(i+1)
            data = np.array(self.metrics[metric])
            plt.title(metric)
            plt.plot(data)
            plt.show()


def get_logger(file_path):
    logger = logging.getLogger()
    fh = logging.FileHandler(file_path, mode="w", encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setLevel(logging.INFO)
    # ch.setLevel(logging.INFO)

    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class Schedule1():
    def __init__(self, optimizer, logger, decay_rate, patience=0, min_mode=True):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.patience = patience
        self.pre_loss = None
        self.no_improve_num = 0
        self.min_mode = min_mode
        self.logger = logger

    def step(self, loss):
        if self.pre_loss is None:
            self.pre_loss = loss
            return

        if self.min_mode:
            if loss >= self.pre_loss:
                self.no_improve_num += 1
            else:
                self.no_improve_num = 0
        else:
            if loss <= self.pre_loss:
                self.no_improve_num += 1
            else:
                self.no_improve_num = 0

        if self.no_improve_num > self.patience:
            lr = self.optimizer.param_groups[0]["lr"] * self.decay_rate
            self.logger.info("not improved decay lr to:  %.10f" % lr)
            self.optimizer.param_groups[0]["lr"] = lr
            self.no_improve_num = 0
        self.pre_loss = loss

    def reset(self, optimizer, if_changed):
        if if_changed:
            self.logger.info("Reset Schedule")
            self.optimizer = optimizer
            self.pre_loss = None
            self.no_improve_num = 0


def tensor2text(text_ids_tensor, vocab):
    text_ids_list = text_ids_tensor.tolist()  # (batch, length)

    result_list = []
    for ids_seq in text_ids_list:
        text_list = [vocab.id2word[item] for item in ids_seq if item != vocab(PAD_TOKEN)]
        result_list.append(text_list)
    return result_list


def remove_all(model_name, optim_name, out_dir):
    model_name = remove_replace(model_name)
    optim_name = remove_replace(optim_name)
    model_name = os.path.join(out_dir, model_name)
    optimizer_name = os.path.join(out_dir, optim_name)
    os.system("rm %s"%model_name)
    os.system("rm %s"%optimizer_name)


def remove_replace(name_str):
    replaced_str = name_str.replace("{:0>8d}", "*")
    return replaced_str


def config_backup(file_name, out_dir):
    os.system("cp %s %s" % (file_name, os.path.join(out_dir, "config_save.txt")))


def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)


class DistLogger():
    def __init__(self, logger, rank=0):
        self.logger = logger
        self.rank = rank

    def info(self, info_str):
        if self.rank == 0:
            self.logger.info(info_str)
        else:
            pass

