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
from prophet.stage1.utils.load_data import CommonData, DataSet
from prophet.stage1.utils.optim import get_optim_for_finetune as get_optim
from timm.models import create_model

from beit3.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner, get_is_head_flag_for_vit
from prophet.stage1.utils.optim import get_optim_for_finetune as get_optim
from beit3.utils import NativeScalerWithGradNormCount as NativeScaler
from beit3.engine_for_finetuning import train_one_epoch, evaluate, VQAHandler
from beit3.datasets import create_downstream_dataset
import beit3.modeling_finetune
import beit3.utils as utils
from beit3.utils import save_on_master


device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def eval(args):

    # Load BEiT3
    model = create_model(
        'beit3_base_patch16_224_okvqa',
        pretrained=False,
        drop_path_rate=0.15,
        #   vocab_size=4477,
        checkpoint_activations='store_true',
            )
    
    # Load checkpoint        
    print('Loading ckpt {}'.format(args.ckpt_path))
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print('Finish!')

    # Load dataloader 
    data_loader_train, data_loader_val = create_downstream_dataset()
    print(f'Length of train data loader: {len(data_loader_train)}, valid data loader: {len(data_loader_val)}')

    task_handler = VQAHandler()

    predictions, _ = evaluate(data_loader_val, model, device, task_handler)
    exit() # score만 계산 

    prediction_file = utils.dump_predictions(args, predictions, f"{args.task}_val_e{epoch}")
    result_file = os.path.join(args.output_dir, f"{args.task}_result_val_e{epoch}.json")
    task_key = "CIDEr"

    test_stats = utils.coco_caption_eval(args.output_dir, prediction_file, "{}_val".format(okvqa))
    utils.write_result_to_jsonl(test_stats, result_file)

    print(f"Performance of the network on the {len(data_loader_val.dataset)} val images: {test_stats[task_key]:.1f}%")
    # if max_accuracy < test_stats[task_key]:
    #     max_accuracy = test_stats[task_key]

    # print(f'Max performance: {max_accuracy:.2f}%')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_cache_path', type=str, default='', help='')
    parser.add_argument('--task', type=str, default='okvqa', help='')
    parser.add_argument('--output_dir', type=str, default='', help='')
    parser.add_argument('--ckpt_path', type=str, default='', help='')
    args = parser.parse_args()

    eval(args)