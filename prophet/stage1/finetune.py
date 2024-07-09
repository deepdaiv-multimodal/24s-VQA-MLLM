import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .utils.load_data import CommonData, DataSet
from .model.beit3 import BEiT3ForVisualQuestionAnswering
# import deepspeed
import utils
import os, sys
from timm.optim.lookahead import Lookahead
# os.environ['TORCH_USE_CUDA_DSA'] = '0'
# sys.path.append(os.getcwd())
from timm.models import create_model
from datetime import datetime
import pickle, random, math, time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
from pathlib import Path
from copy import deepcopy
import yaml
from beit3.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner, get_is_head_flag_for_vit
from .utils.optim import get_optim_for_finetune as get_optim
from beit3.utils import NativeScalerWithGradNormCount as NativeScaler
from beit3.engine_for_finetuning import train_one_epoch, evaluate, VQAHandler
from beit3.datasets import create_downstream_dataset
import beit3.modeling_finetune
import beit3.utils as utils
from beit3.utils import save_on_master
# Define args directly in the script
class Args:
    def __init__(self):
        self.task = 'vqav2'
        self.input_size = 480
        self.drop_path = 0.15
        # self.drop_path = 0.1
        self.checkpoint_activations = False
        self.sentencepiece_model = '/root/datasets/okvqa/data/beit3.spm'
        self.vocab_size = 64010
        self.num_max_bpe_tokens = 64
        self.model_ema = False
        self.model_ema_decay = 0.9999
        self.model_ema_force_cpu = False
        self.opt = 'adamw'
        self.opt_eps = 1e-8
        self.opt_betas = [0.9, 0.98]
        # self.clip_grad = Nonelr
        self.momentum = 0.9
        self.weight_decay = 0.01
        self.lr = 2e-5
        # self.lr = 3e-5
        self.layer_decay = 1.0
        self.task_head_lr_weight = 20
        self.warmup_lr = 1e-6
        self.min_lr = 1e-6
        self.warmup_epochs = 1
        self.warmup_steps = -1
        self.batch_size = 32
        self.eval_batch_size = 1
        self.epochs = 100
        self.update_freq = 1
        self.save_ckpt_freq = 5
        self.randaug = True
        self.train_interpolation = 'bicubic'
        self.finetune = ''
        self.model_key = 'model|module'
        self.model_prefix = ''
        self.data_path = '/root/datasets/okvqa/data'
        self.output_dir = '/root/datasets/okvqa/data/beit3_large_ckpt'
        self.log_dir = None
        self.device = 'cuda'
        self.seed = 0
        self.resume = ''
        self.auto_resume = True
        self.save_ckpt = True
        self.start_epoch = 0
        self.eval = False
        self.dist_eval = False
        self.num_workers = 4
        self.pin_mem = True
        self.world_size = 1
        self.local_rank = -1
        self.dist_on_itp = False
        self.dist_url = 'env://'
        self.task_cache_path = '/root/datasets/okvqa/data/beit3_large_ckpt'
        self.nb_classes = 1000
        self.mixup = 0
        self.cutmix = 0
        self.cutmix_minmax = None
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = 'batch'
        self.color_jitter = 0.4
        self.aa = 'rand-m9-mstd0.5-inc1'
        self.smoothing = 0.1
        self.crop_pct = None
        self.reprob = 0.25
        self.remode = 'pixel'
        self.recount = 1
        self.resplit = False
        self.captioning_mask_prob = 0.6
        self.drop_worst_ratio = 0.2
        self.drop_worst_after = 12000
        self.num_beams = 3
        self.length_penalty = 0.6
        self.label_smoothing = 0.1
        self.enable_deepspeed = False
        self.initial_scale_power = 16
        self.zero_stage = 0

args = Args()

def save_model(output_dir, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)

def create_optimizer(model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = 'adamw'
    weight_decay = 0.05
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=5e-4, weight_decay=weight_decay)
    opt_args['eps'] = 1e-8
    opt_args['betas'] = [0.9, 0.999]

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError("Invalid optimizer")

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer

class Runner:
    def __init__(self, __C, evaluator):
        self.__C = __C
        self.evaluator = evaluator

    def train(self, train_set, eval_set=None):
        data_loader_train, data_loader_val = create_downstream_dataset()
        data_size = train_set.data_size

        model = create_model(
                  'beit3_base_patch16_224_okvqa',
                  pretrained=False,
                  drop_path_rate=args.drop_path,
                #   vocab_size=4477,
                  checkpoint_activations='store_true',
              )
              
        # utils.load_model_and_may_interpolate('/root/datasets/okvqa/data/beit3_base_patch16_224.pth', model, 'model|module', '')
        utils.load_model_and_may_interpolate('/root/datasets/okvqa/data/beit3_large_patch16_224.pth', model, 'model|module', '')

        model.to("cuda")

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size

        num_layers = model.get_num_layers()
        lrs = list(0.9 ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)

        skip_weight_decay_list = model.no_weight_decay()

        # optimizer 
        optimizer = create_optimizer(
            model, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)

        loss_scaler = NativeScaler()

        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )   

        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=args.model_ema)


        task_handler = VQAHandler()


        max_accuracy = 0.0

        for epoch in range(100):
          train_stats = train_one_epoch(
              model, data_loader_train, optimizer, "cuda", task_handler, epoch, 
              epoch * num_training_steps_per_epoch, lr_schedule_values, loss_scaler, 
              None, 1, None, None, 'vqav2', None,
          )

          if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs:
              save_model(output_dir=args.output_dir, epoch=epoch, model=model, model_without_ddp=model, optimizer=optimizer,
                  loss_scaler=loss_scaler, model_ema=None)
                  
        #   if data_loader_val is not None:
        #       predictions, _ = evaluate(data_loader_val, model, "cuda", task_handler)
        #       if utils.is_main_process():
        #           test_stats = utils.coco_caption_eval(args.output_dir, prediction_file, "{}_val".format(vqav2))
        #           utils.write_result_to_jsonl(test_stats, result_file)

        #       torch.distributed.barrier()
        #       if not utils.is_main_process():
        #           test_stats = utils.read_result_from_jsonl(result_file)

        #       print(f"Performance of the network on the {len(data_loader_val.dataset)} val images: {test_stats[task_key]:.1f}%")
        #       if max_accuracy < test_stats[task_key]:
        #           max_accuracy = test_stats[task_key]

        #       print(f'Max performance: {max_accuracy:.2f}%')
              
              log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        #   **{f'val_{k}': v for k, v in test_stats.items()},
                          'epoch': epoch,
                          'n_parameters': n_parameters}

        # net = BEiT3ForVisualQuestionAnswering(self.__C, num_classes=train_set.ans_size)

        # # Define the optimizer
        # if self.__C.RESUME:
        #     raise NotImplementedError('Resume training is not needed as the finetuning is fast')
        # else:
        #     optim = get_optim(self.__C, net)
        #     start_epoch = 0

        # # load to gpu
        # net.cuda()
        # # Define the multi-gpu training if needed
        # if self.__C.N_GPU > 1:
        #     net = nn.DataParallel(net, device_ids=self.__C.GPU_IDS)

        # # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        # epoch_loss = 0

        # # Define multi-thread dataloader
        # dataloader = Data.DataLoader(
        #     train_set,
        #     batch_size=self.__C.BATCH_SIZE,
        #     shuffle=True,
        #     num_workers=self.__C.NUM_WORKERS,
        #     pin_memory=self.__C.PIN_MEM,
        #     drop_last=True
        # )

        # # Training script
        # for epoch in range(start_epoch, self.__C.MAX_EPOCH):
        #     net.train()
        #     # Save log information
        #     with open(self.__C.LOG_PATH, 'a+') as logfile:
        #         logfile.write(
        #             f'nowTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n'
        #         )

        #     time_start = time.time()

        #     # Iteration
        #     for step, input_tuple in enumerate(dataloader):
        #         iteration_loss = 0
        #         optim.zero_grad()
        #         input_tuple = [x.cuda() for x in input_tuple]
        #         SUB_BATCH_SIZE = self.__C.BATCH_SIZE // self.__C.GRAD_ACCU_STEPS
        #         for accu_step in range(self.__C.GRAD_ACCU_STEPS):

        #             sub_tuple = [x[accu_step * SUB_BATCH_SIZE:
        #                 (accu_step + 1) * SUB_BATCH_SIZE] for x in input_tuple]
                    
        #             sub_ans_iter = sub_tuple[-1]
        #             pred = net(sub_tuple[:-1])
        #             loss = loss_fn(pred, sub_ans_iter)
        #             loss.backward()
        #             loss_item = loss.item()
        #             iteration_loss += loss_item
        #             epoch_loss += loss_item# * self.__C.GRAD_ACCU_STEPS

        #         print("\r[version %s][epoch %2d][step %4d/%4d][Task %s][Mode %s] loss: %.4f, lr: %.2e" % (
        #             self.__C.VERSION,
        #             epoch + 1,
        #             step,
        #             int(data_size / self.__C.BATCH_SIZE),
        #             self.__C.TASK,
        #             self.__C.RUN_MODE,
        #             iteration_loss / self.__C.BATCH_SIZE,
        #             optim.current_lr(),
        #         ), end='          ')

        #         optim.step()

        #     time_end = time.time()
        #     print('Finished in {}s'.format(int(time_end - time_start)))

        #     # Logging
        #     with open(self.__C.LOG_PATH, 'a+') as logfile:
        #         logfile.write(f'epoch = {epoch + 1}  loss = {epoch_loss / data_size}\nlr = {optim.current_lr()}\n\n')
            
        #     optim.schedule_step(epoch)

        #     # Save checkpoint
        #     state = {
        #         'state_dict': net.state_dict() if self.__C.N_GPU == 1 \
        #             else net.module.state_dict(),
        #         'optimizer': optim.optimizer.state_dict(),
        #         'warmup_lr_scale': optim.warmup_lr_scale,
        #         'decay_lr_scale': optim.decay_lr_scale,
        #     }
        #     torch.save(
        #         state,
        #         f'{self.__C.CKPTS_DIR}/epoch{epoch + 1}.pkl'
        #     )


        #     # Eval after every epoch
        #     if eval_set is not None:
        #         self.eval(
        #             eval_set,
        #             net,
        #             eval_now=True
        #         )
            
        #     epoch_loss = 0


    def run(self):
        common_data = CommonData(self.__C)
        train_set = DataSet(self.__C, common_data, self.__C.TRAIN_SPLITS)
        valid_set = DataSet(self.__C, common_data, self.__C.EVAL_SPLITS)
        self.train(train_set, valid_set)