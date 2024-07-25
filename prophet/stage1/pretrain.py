import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime
import pickle
import random
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import argparse
from pathlib import Path
import yaml

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet
from .model.beit3 import BEiT3_VQA
from .utils.optim import get_optim_for_finetune as get_optim
from evaluation.okvqa_evaluate import OKEvaluater

class Runner(object):
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater

    def train(self, train_set, eval_set=None, evaluater=None):
        # Set paths and parameters directly
        model_name_or_path = "microsoft/beit-base-patch16-224-in21k"
        answer_size = train_set.ans_size

        net = BEiT3_VQA(
            model_name_or_path=model_name_or_path,
            answer_size=answer_size
        )

        optim = get_optim(self.__C, net)  # Configure optimizer
        net.cuda()
        net = nn.DataParallel(net, device_ids=self.__C.GPU_IDS) if self.__C.N_GPU > 1 else net
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        dataloader = Data.DataLoader(train_set, batch_size=self.__C.BATCH_SIZE, shuffle=True, num_workers=self.__C.NUM_WORKERS, pin_memory=self.__C.PIN_MEM, drop_last=True)
        
        for epoch in range(self.__C.MAX_EPOCH):
            net.train()
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(f'nowTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n')
            time_start = time.time()
            epoch_loss = 0
            
            for step, input_tuple in enumerate(dataloader):
                optim.zero_grad()
                input_tuple = [x.cuda() for x in input_tuple]
                pred = net(input_tuple[0], input_tuple[1])
                loss = loss_fn(pred, input_tuple[2])
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                
                print(f"\r[version {self.__C.VERSION}][epoch {epoch+1}][step {step}/{int(len(train_set) / self.__C.BATCH_SIZE)}] loss: {epoch_loss / (step+1):.4f}, lr: {optim.current_lr():.2e}", end=' ')
            
            time_end = time.time()
            print('Finished in {}s'.format(int(time_end - time_start)))
            
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(f'epoch = {epoch + 1}  loss = {epoch_loss / len(train_set)}\nlr = {optim.current_lr()}\n\n')
            
            optim.schedule_step(epoch)
            self.save_checkpoint(net, optim, epoch)
            
            if eval_set and evaluater:
                self.eval(eval_set, net, evaluater)

    def save_checkpoint(self, net, optim, epoch):
        state = {'state_dict': net.state_dict(), 'optimizer': optim.state_dict()}
        torch.save(state, f'{self.__C.CKPTS_DIR}/epoch{epoch + 1}.pkl')

    @torch.no_grad()
    def eval(self, dataset, net=None, evaluater=None):
        if net is None:
            model_name_or_path = "microsoft/beit-base-patch16-224-in21k"
            answer_size = dataset.ans_size

            net = BEiT3_VQA(
                model_name_or_path=model_name_or_path,
                answer_size=answer_size
            )

            checkpoint = torch.load(self.__C.CKPT_PATH)
            net.load_state_dict(checkpoint['state_dict'])
            net.cuda()
            net = nn.DataParallel(net, device_ids=self.__C.GPU_IDS) if self.__C.N_GPU > 1 else net
        
        net.eval()
        dataloader = Data.DataLoader(dataset, batch_size=self.__C.EVAL_BATCH_SIZE, shuffle=False, num_workers=self.__C.NUM_WORKERS, pin_memory=True)
        qid_idx = 0
        evaluater.init()
        
        for step, input_tuple in enumerate(dataloader):
            input_tuple = [x.cuda() for x in input_tuple]
            pred = net(input_tuple[0], input_tuple[1])
            pred_np = pred.cpu().numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            
            for i in range(len(pred_argmax)):
                qid = dataset.qids[qid_idx]
                qid_idx += 1
                ans_id = int(pred_argmax[i])
                ans = dataset.ix_to_ans[ans_id]
                evaluater.add(qid, ans)
        
        evaluater.save(self.__C.RESULT_PATH)
        
        with open(self.__C.LOG_PATH, 'a+') as logfile:
            evaluater.evaluate(logfile)

    def run(self):
        Path(self.__C.CKPTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(self.__C.RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        
        common_data = CommonData(self.__C)
        
        if self.__C.RUN_MODE == 'finetune':
            train_set = DataSet(self.__C, common_data, self.__C.TRAIN_SPLITS)
            valid_set = DataSet(self.__C, common_data, self.__C.EVAL_SPLITS) if self.__C.EVAL_NOW else None
            evaluater = OKEvaluater(
                self.__C.EVAL_ANSWER_PATH,
                self.__C.EVAL_QUESTION_PATH,
            )
            self.train(train_set, valid_set, evaluater)
        
        elif self.__C.RUN_MODE == 'finetune_test':
            test_set = DataSet(self.__C, common_data, self.__C.EVAL_SPLITS)
            evaluater = OKEvaluater(
                self.__C.EVAL_ANSWER_PATH,
                self.__C.EVAL_QUESTION_PATH,
            )
            self.eval(test_set, None, evaluater)
        
        else:
            raise ValueError('Invalid run mode')

def finetune_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--run_mode', dest='RUN_MODE', help='run mode: finetune/finetune_test', type=str, required=True)
    parser.add_argument('--cfg_file', dest='CFG_FILE', help='path to the config file', type=str, required=True)
    parser.add_argument('--n_gpu', dest='N_GPU', help='number of gpus', type=int, default=1)
    parser.add_argument('--gpu_ids', dest='GPU_IDS', help='gpu ids to use', type=str, default='0')
    parser.add_argument('--batch_size', dest='BATCH_SIZE', help='batch size', type=int, default=64)
    parser.add_argument('--eval_batch_size', dest='EVAL_BATCH_SIZE', help='eval batch size', type=int, default=256)
    parser.add_argument('--max_epoch', dest='MAX_EPOCH', help='max epoch', type=int, default=60)
    parser.add_argument('--seed', dest='SEED', help='random seed', type=int, default=123)
    parser.add_argument('--num_workers', dest='NUM_WORKERS', help='num_workers', type=int, default=4)
    parser.add_argument('--pin_mem', dest='PIN_MEM', help='pin memory', action='store_true', default=True)
    parser.add_argument('--log_path', dest='LOG_PATH', help='log path', type=str, default='result/log')
    parser.add_argument('--ckpts_dir', dest='CKPTS_DIR', help='checkpoints dir', type=str, default='result/ckpts')
    parser.add_argument('--result_path', dest='RESULT_PATH', help='result path', type=str, default='result/result.json')
    parser.add_argument('--eval_now', dest='EVAL_NOW', help='eval now', type=bool, default=False)
    parser.add_argument('--resume_version', dest='RESUME_VERSION', help='checkpoint version name', type=str, default='')
    parser.add_argument('--resume_epoch', dest='RESUME_EPOCH', help='checkpoint epoch', type=int, default=1)
    parser.add_argument('--resume_path', dest='RESUME_PATH', help='checkpoint path', type=str, default='')
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for pretraining')
    finetune_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    
    evaluater = OKEvaluater(
        __C.EVAL_ANSWER_PATH,
        __C.EVAL_QUESTION_PATH,
    )
    
    runner = Runner(__C, evaluater)
    runner.run()
