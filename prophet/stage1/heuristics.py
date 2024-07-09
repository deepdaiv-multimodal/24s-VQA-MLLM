import os
import sys
from datetime import datetime
import pickle
import random
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as Data
import argparse
from pathlib import Path
import yaml
from copy import deepcopy
from tqdm import tqdm

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet
from .model.beit3 import BEiT3ForVisualQuestionAnswering
from .utils.optim import get_optim_for_finetune as get_optim
from timm.models import create_model

from beit3.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner, get_is_head_flag_for_vit
from .utils.optim import get_optim_for_finetune as get_optim
from beit3.utils import NativeScalerWithGradNormCount as NativeScaler
from beit3.engine_for_finetuning import train_one_epoch, evaluate, VQAHandler
from beit3.datasets import create_downstream_dataset
import beit3.modeling_finetune
import beit3.utils as utils
from beit3.utils import save_on_master

device = "cuda" if torch.cuda.is_available() else "cpu"

class Runner(object):
    def __init__(self, __C, *args, **kwargs):
        self.__C = __C
        self.net = None

    # heuristics generation
    @torch.no_grad()
    def eval(self, data_loader):
        # data_size = dataset.data_size

        # if self.net is None:
        #     # Load parameters
        #     path = self.__C.CKPT_PATH
        #     print('Loading ckpt {}'.format(path))
        #     net = BEiT3ForVisualQuestionAnswering(self.__C, num_classes=dataset.ans_size)
        #     ckpt = torch.load(path, map_location='cpu')
        #     net.load_state_dict(ckpt, strict=False)
        #     net.half()  # Model parameters to half precision
        #     net.cuda()
        #     if self.__C.N_GPU > 1:
        #         net = nn.DataParallel(net, device_ids=self.__C.GPU_IDS)
        #     print('Finish!')
        #     self.net = net
        # else:
        #     net = self.net

        # Load BEiT3
        model = create_model(
            'beit3_base_patch16_224_okvqa',
            pretrained=False,
            drop_path_rate=0.1,
            #   vocab_size=4477,
            checkpoint_activations='store_true',
              )
              
        path = self.__C.CKPT_PATH
        print('Loading ckpt {}'.format(path))
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        # if 'optimizer' in checkpoint and 'epoch' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     args.start_epoch = checkpoint['epoch'] + 1
        #     if hasattr(args, 'model_ema') and args.model_ema:
        #         _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        #     if 'scaler' in checkpoint:
        #         loss_scaler.load_state_dict(checkpoint['scaler'])
        #     print("With optim & sched!")
        model.to(device)
        print('Finish!')
        self.model = model

        model.eval()
        
        # dataloader = Data.DataLoader(
        #     dataset,
        #     batch_size=1,  # Batch size set to 1
        #     shuffle=False,
        #     num_workers=self.__C.NUM_WORKERS,
        #     pin_memory=True
        # )
        

        qid_idx = 0
        topk_results = {}
        latent_results = []
        k = self.__C.CANDIDATE_NUM

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        # switch to evaluation mode
        # handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)
        que_ids = []
        for data in metric_logger.log_every(data_loader, 10, header):
            for tensor_key in data.keys():
                data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                # handler.eval_batch(model=model, **data)
                pred, answer_latents = model(
                                    image=data['image'], question=data['language_tokens'], 
                                    padding_mask=data['padding_mask'])
                # batch_size = language_tokens.shape[0]

                bs = pred.shape[0]
                pred = pred.view(bs, -1)

                pred_np = pred.sigmoid().cpu().numpy() # 2910
                answer_latents_np = answer_latents.cpu().numpy() if answer_latents is not None else None
                # if labels is not None:
                #     scores = utils.VQAScore()(logits, labels) * 100.0
                #     self.metric_logger.meters['score'].update(scores.item(), n=batch_size)
                # else:
                #     _, preds = logits.max(-1)
                #     for image_id, pred in zip(qid, preds):
                #         self.predictions.append({
                #             "question_id": image_id.item(), 
                #             "answer": self.label2ans[pred.item()], 
                #         })

                 # Check the shape of pred_np
                if len(pred_np.shape) == 1:
                    pred_np = pred_np[np.newaxis, :]

                qid = data['qid'].item()

                ans_np = pred_np.flatten()
                ans_idx = np.argsort(-ans_np)[:k]
                # print(ans_idx)
                # exit()

                ans_item = []

                for idx in ans_idx:
                    ans_item.append(
                        {
                            'answer': data_loader.dataset.label2ans[idx],
                            'confidence': float(ans_np[idx])
                        }
                    )
                topk_results[qid] = ans_item

                # print('answer_latents_np', answer_latents_np.shape)
                # exit()
                if answer_latents_np is not None:
                    latent_np = answer_latents_np
                    latent_results.append(latent_np)
                    # np.save(
                    #     os.path.join(self.__C.ANSWER_LATENTS_DIR, f'{qid}.npy'),
                    #     latent_np
                    # )

        print(f"Total topk results generated: {len(topk_results)}")
        print(f"Total latent results generated: {len(latent_results)}")
        
        return topk_results, latent_results


    def run(self):
        # Set ckpts and log path
        ## where checkpoints will be saved
        Path(self.__C.CKPTS_DIR).mkdir(parents=True, exist_ok=True)
        ## where the result file of topk candidates will be saved
        Path(self.__C.CANDIDATE_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
        ## where answer latents will be saved
        Path(self.__C.ANSWER_LATENTS_DIR).mkdir(parents=True, exist_ok=True)

        # build dataset entities        
        # common_data = CommonData(self.__C)
        # train_set = DataSet(
        #     self.__C,
        #     common_data,
        #     self.__C.TRAIN_SPLITS
        # )

        # test_set = DataSet(
        #     self.__C,
        #     common_data,
        #     self.__C.EVAL_SPLITS
        # )

        # print(f"Total training questions: {len(train_set.qids)}")
        # print(f"Total testing questions: {len(test_set.qids)}")

        data_loader_train, data_loader_val = create_downstream_dataset()
        print(f'Load {len(data_loader_train)} of tain data, {len(data_loader_val)} of valid data')

        # forward VQA model
        train_topk_results, train_latent_results = self.eval(data_loader_train)
        test_topk_results, test_latent_results = self.eval(data_loader_val)

        # save topk candidates
        # topk_results = {**train_topk_results, **test_topk_results}

        # json.dump(
        #     topk_results,
        #     open(self.__C.CANDIDATE_FILE_PATH, 'w'),
        #     indent=4
        # )

        # print(f"Total topk results saved: {len(topk_results)}")
        print(f"Total topk results saved: {len(test_topk_results)}")


        # search similar examples
        train_features = np.vstack(train_latent_results)
        train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
        
        print('test_latent_results', len(test_latent_results), test_latent_results[0].shape)
        test_features = np.vstack(test_latent_results)
        test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)

        # Check lengths of test and train features
        # min_len = min(len(test_features), len(train_features))
        # min_len = len(test_features)
        # print(min_len)
        print('test_features: ', test_features.shape)
        print('train_features: ', train_features.shape)
        
        # train_features = train_features.reshape(train_features.shape[0],)
        train_features = train_features[:, 0, :]
        test_features = test_features[:, 0, :]
        print('test_features: ', test_features.shape)
        print('train_features: ', train_features.shape)

        # compute top-E similar examples for each testing input
        E = self.__C.EXAMPLE_NUM
        similar_qids = {}

        print(f'\ncompute top-{E} similar examples for each testing input')

        # for i, test_qid in enumerate(tqdm(test_set.qids[:min_len])):
        #     # cosine similarity
        #     dists = np.dot(test_features[i], train_features.T)
        #     top_E = np.argsort(-dists)[:E]
        #     similar_qids[test_qid] = [train_set.qids[j] for j in top_E
        # print(data_loader_val.dataset.items[:min_len])
        
        
        for i, test_qid in enumerate(tqdm([item["qid"] for item in data_loader_val.dataset.items])):
            dists = np.dot(test_features[i], train_features.T)
            top_E = np.argsort(-dists)[:E]
            print('top_E', top_E)
            similar_qids[test_qid] = [data_loader_train.dataset.items[j]["qid"] for j in top_E]
            print('similar_qids: ', similar_qids)

        # save similar qids
        with open(self.__C.EXAMPLE_FILE_PATH, 'w') as f:
            json.dump(similar_qids, f)
        print(f"Total similar qids saved: {len(similar_qids)}")

        # for val_data in metric_logger.log_every(data_loader_val, 10, header):
        #     qid = data_loader_val['qid']

            # for tensor_key in data.keys():
            #     data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

def heuristics_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for heuristics', type=str, default=None)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--candidate_num', dest='CANDIDATE_NUM', help='topk candidates', type=int, default=None)
    parser.add_argument('--example_num', dest='EXAMPLE_NUM', help='number of most similar examples to be searched, default: 200', type=int, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for pretraining')
    heuristics_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)
    runner = Runner(__C)
    runner.run()