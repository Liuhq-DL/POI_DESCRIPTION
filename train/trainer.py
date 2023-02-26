import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import train_config, model_config
import math
from config.CONSTANTS import *
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer():
    def __init__(self, model, criterion, device, data_processor, logger, resume=-1, distributed=False, rank=0,
                 parallel=False):
        self.device = device
        self.distributed = distributed
        self.rank = rank
        self.parallel = parallel
        if parallel:
            print("Warrning: some ability can't use")
            model = torch.nn.DataParallel(model)  #  device_ids=device, output_device=cuda0
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.data_processor = data_processor
        self.args = train_config
        self.logger = logger
        self.update_interval = train_config["update_interval"]
        self.org_model = self.model
        self.curr_iters = 0

        self.use_pre_train = model_config["use_pretrained"] if "use_pretrained" in model_config else False

        self.set_optimizer(lr=train_config["init_lr"])

        if resume>0:
            self.load_checkpoint(resume, load_optim=train_config["save_optim"])
            print("resume to train")
            if self.use_pre_train:
                raise RuntimeError

        if distributed:
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

        self.epoch = 0
        self.iteration = 0

    def set_optimizer(self, lr):
        self.grad_parameters = list(filter(lambda p: p.requires_grad, self.org_model.parameters()))
        if train_config["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(params=self.grad_parameters, lr=lr,
                                              **train_config["Adam"])
        elif train_config["optimizer"] == "AdamW":
            self.optimizer = torch.optim.AdamW(params=self.grad_parameters, lr=lr,
                                               **train_config["AdamW"])
        else:
            self.optimizer = torch.optim.SGD(params=self.grad_parameters, lr=lr)

    def get_train_info(self, epoch, iteration):
        changed, stage = self.data_processor.switch_stage(self.org_model, epoch, iteration)
        if changed:
            this_lr = train_config["stage_lr_list"][stage]
            print("changed this lr: ", this_lr)
            self.set_optimizer(this_lr)

        self.epoch = epoch
        self.iteration = iteration
        return changed

    def train_one_batch(self, data):
        data_processed = self.data_processor(data)
        model_input = self.data_processor.data_to_forward(data_processed)

        self.model.train()

        model_output = self.model(*model_input)
        loss_target = self.data_processor.data_output_to_criterion(data_processed, model_output)
        loss = self.criterion(*loss_target)

        loss.backward()

        grad_norm = "wait"
        self.curr_iters += 1
        if self.curr_iters % self.update_interval == 0:
            if self.args["clip_norm"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.grad_parameters, self.args["clip_norm"])
            elif self.args["clip_value"] > 0:
                torch.nn.utils.clip_grad_value_(self.grad_parameters, self.args["clip_value"])
                grad_norm = math.sqrt(sum(p.grad.data.norm() ** 2 for p in self.grad_parameters))
            else:
                grad_norm = math.sqrt(sum(p.grad.data.norm() ** 2 for p in self.grad_parameters))
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.curr_iters = 0
        refer_token_sum = self.data_processor.data_to_token_sum(data_processed)

        return loss.item(), refer_token_sum, grad_norm

    def valid_one_batch(self, data):
        data_processed = self.data_processor(data)
        model_input = self.data_processor.data_to_forward(data_processed)

        self.model.eval()
        refer_token_sum = self.data_processor.data_to_token_sum(data_processed)
        with torch.no_grad():
            model_output = self.model(*model_input)
            loss_target = self.data_processor.data_output_to_criterion(data_processed, model_output)
            loss = self.criterion(*loss_target)

        return loss.item(), refer_token_sum

    def save_checkpoint(self, iteration, save_optim=False):
        model_state_dict = self.model.state_dict() if not (self.distributed or self.parallel)\
            else self.model.module.state_dict()
        torch.save(model_state_dict,
                   os.path.join(self.args["out_dir"], MODEL_NAME.format(iteration)))
        if save_optim:
            torch.save(self.optimizer.state_dict(),
                   os.path.join(self.args["out_dir"], OPTIMIZER_NAME.format(iteration)))

    def load_checkpoint(self, iteration, load_optim=False):
        if self.distributed:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        else:
            map_location = None
        self.model.load_state_dict(torch.load(os.path.join(self.args["out_dir"],
                                                           MODEL_NAME.format(iteration)),
                                              map_location=map_location))

        if load_optim:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.args["out_dir"],
                                                           OPTIMIZER_NAME.format(iteration))))

    def del_check_point(self, iteration, save_optim=False):
        os.system("rm %s"%os.path.join(self.args["out_dir"], MODEL_NAME.format(iteration)))
        if save_optim:
            os.system("rm %s"% os.path.join(self.args["out_dir"], OPTIMIZER_NAME.format(iteration)))





