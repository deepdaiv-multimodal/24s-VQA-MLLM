import os
import random
import numpy as np
import torch
from datetime import datetime
import yaml
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from .task_to_split import *
from .path_cfgs import PATH

class Cfgs(PATH):
    def __init__(self, args):
        super(Cfgs, self).__init__()
        self.set_silent_attr()

        self.GPU = getattr(args, 'GPU', None)
        if self.GPU is not None:
            self.GPU_IDS = [int(i) for i in self.GPU.split(',')]
            self.CURRENT_GPU = self.GPU_IDS[0]
            torch.cuda.set_device(f'cuda:{self.CURRENT_GPU}')
            self.N_GPU = len(self.GPU_IDS)
            self.SEED = getattr(args, 'SEED', 1111)
            torch.manual_seed(self.SEED)
            if self.N_GPU < 2:
                torch.cuda.manual_seed(self.SEED)
            else:
                torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            np.random.seed(self.SEED)
            random.seed(self.SEED)
            torch.set_num_threads(2)

        # 기본값 설정
        self.deepspeed_config = '/content/drive/MyDrive/prophet/deepspeed_config.json'
        self.vocab_size = 64010
        self.opt_betas = (0.9, 0.999)
        self.encoder_embed_dim = 768
        self.img_size = 224
        self.patch_size = 16
        self.drop_path_rate = 0.1
        self.checkpoint_activations = False
        self.multiway = True
        self.share_encoder_input_output_embed = False
        self.in_chans = 3
        self.max_source_positions = 1024
        self.dropout = 0.1
        self.no_scale_embedding = False
        self.no_output_layer = False
        self.layernorm_embedding = True
        self.layernorm_eps = 1e-5
        self.moe_freq = 0
        self.moe_expert_count = 1
        self.moe_gating_use_fp32 = False
        self.moe_second_expert_policy = 'all'
        self.moe_normalize_gate_prob_before_dropping = False
        self.moe_expert_dropout = 0.0
        self.moe_gate_loss_wt = 1.0
        self.moe_gate_loss_combine_method = 'sum'
        self.moe_gate_loss_rank = 0
        self.moe_eom_loss_wt = 0.0
        self.encoder_layers = 12
        self.encoder_attention_heads = 12
        self.attention_dropout = 0.1
        self.activation_dropout = 0.1
        self.subln = False
        self.xpos_rel_pos = False
        self.encoder_normalize_before = True
        self.encoder_ffn_embed_dim = 3072
        self.activation_fn = 'relu'
        self.deepnorm = False
        self.fsdp = False
        self.normalize_output = False
        self.rel_pos_buckets = 32
        self.max_rel_pos = 128
        self.bert_init = False

        # 추가된 속성
        self.pretrained_model_path = '/root/datasets/okvqa/data/beit3_large_patch16_224.pth'
        self.lr_base = 5e-5
        self.weight_decay = 0.01
        self.epochs = 10
        self.batch_size = 64

        self.TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')
        self.VERSION = getattr(args, 'VERSION', self.TIMESTAMP)
        self.CKPTS_DIR = os.path.join(self.CKPT_ROOT, self.VERSION)
        self.LOG_PATH = os.path.join(self.LOG_ROOT, self.VERSION, f'log_{self.TIMESTAMP}.txt')
        self.RESULT_DIR = os.path.join(self.RESULTS_ROOT, self.VERSION)
        self.RESULT_PATH = os.path.join(self.RESULTS_ROOT, self.VERSION, 'result_' + self.TIMESTAMP + '.json')

        self.RESUME = getattr(args, 'RESUME', False)
        if self.RESUME and self.RUN_MODE == 'pretrain':
            self.RESUME_VERSION = getattr(args, 'RESUME_VERSION', self.VERSION)
            self.RESUME_EPOCH = getattr(args, 'RESUME_EPOCH', None)
            resume_path = getattr(args, 'RESUME_PATH', None)
            self.RESUME_PATH = os.path.join(self.CKPTS_DIR, self.RESUME_VERSION, f'epoch_{self.RESUME_EPOCH}.pkl') if resume_path is None else resume_path

        self.CKPT_PATH = getattr(args, 'CKPT_PATH', None)

        self.TASK = getattr(args, 'TASK', 'ok')
        assert self.TASK in ['ok', 'aok_val', 'aok_test']

        self.RUN_MODE = getattr(args, 'RUN_MODE', 'finetune')
        assert self.RUN_MODE in ['pretrain', 'finetune', 'finetune_test', 'heuristics', 'prompt']

        if self.RUN_MODE == 'pretrain':
            self.DATA_TAG = 'v2'
            self.DATA_MODE = 'pretrain'
        else:
            self.DATA_TAG = self.TASK.split('_')[0]
            self.DATA_MODE = 'finetune'

        self.EVAL_NOW = True
        if self.RUN_MODE == 'pretrain' or self.TASK == 'aok_test':
            self.EVAL_NOW = False

        self.NUM_WORKERS = 0
        self.PIN_MEM = True

        self.CANDIDATE_NUM = getattr(args, 'CANDIDATE_NUM', None)
        if self.CANDIDATE_NUM is not None:
            self.CANDIDATE_FILE_PATH = os.path.join(self.RESULTS_ROOT, self.VERSION, 'candidates.json')
            self.EXAMPLE_FILE_PATH = os.path.join(self.RESULTS_ROOT, self.VERSION, 'examples.json')
            self.ANSWER_LATENTS_DIR = os.path.join(self.RESULTS_ROOT, self.VERSION, 'answer_latents')

        for attr in args.__dict__:
            if not hasattr(self, attr):  # 이미 존재하는 속성은 덮어쓰지 않습니다.
                setattr(self, attr, getattr(args, attr))

        self.load_config_from_yaml(args.cfg_file)

    def load_config_from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.override_from_dict(yaml_dict)

    def __repr__(self):
        _str = ''
        for attr in self.__dict__:
            if attr in self.__silent or getattr(self, attr) is None:
                continue
            _str += '{ %-17s }-> %s\n' % (attr, getattr(self, attr))
        return _str

    def override_from_dict(self, dict_):
        for key, value in dict_.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def set_silent_attr(self):
        self.__silent = []
        for attr in self.__dict__:
            self.__silent.append(attr)

    @property
    def TRAIN_SPLITS(self):
        return TASK_TO_SPLIT[self.TASK][self.DATA_MODE]['train_split']

    @property
    def EVAL_SPLITS(self):
        return TASK_TO_SPLIT[self.TASK][self.DATA_MODE]['eval_split']

    @property
    def FEATURE_SPLIT(self):
        FEATURE_SPLIT = []
        for split in self.TRAIN_SPLITS + self.EVAL_SPLITS:
            feat_split = SPLIT_TO_IMGS[split]
            if feat_split not in FEATURE_SPLIT:
                FEATURE_SPLIT.append(feat_split)
        return FEATURE_SPLIT

    @property
    def EVAL_QUESTION_PATH(self):
        return self.QUESTION_PATH[self.EVAL_SPLITS[0]]

    @property
    def EVAL_ANSWER_PATH(self):
        if not self.EVAL_NOW:
            return []
        return self.ANSWER_PATH[self.EVAL_SPLITS[0]]

    @property
    def VOCAB_SIZE(self):
        return self.vocab_size

    @property
    def IMG_SIZE(self):
        return self.img_size

    @property
    def ENCODER_EMBED_DIM(self):
        return self.encoder_embed_dim

    @property
    def PATCH_SIZE(self):
        return self.patch_size

    @property
    def DROP_PATH_RATE(self):
        return self.drop_path_rate

    @property
    def CHECKPOINT_ACTIVATIONS(self):
        return self.checkpoint_activations

    @property
    def MULTIWAY(self):
        return self.multiway

    @property
    def SHARE_ENCODER_INPUT_OUTPUT_EMBED(self):
        return self.share_encoder_input_output_embed

    @property
    def IN_CHANS(self):
        return self.in_chans

    @property
    def MAX_SOURCE_POSITIONS(self):
        return self.max_source_positions

    @property
    def DROPOUT(self):
        return self.dropout

    @property
    def NO_SCALE_EMBEDDING(self):
        return self.no_scale_embedding

    @property
    def NO_OUTPUT_LAYER(self):
        return self.no_output_layer

    @property
    def LAYERNORM_EMBEDDING(self):
        return self.layernorm_embedding

    @property
    def LAYERNORM_EPS(self):
        return self.layernorm_eps

    @property
    def MOE_FREQ(self):
        return self.moe_freq

    @property
    def MOE_EXPERT_COUNT(self):
        return self.moe_expert_count

    @property
    def MOE_GATING_USE_FP32(self):
        return self.moe_gating_use_fp32

    @property
    def MOE_SECOND_EXPERT_POLICY(self):
        return self.moe_second_expert_policy

    @property
    def MOE_NORMALIZE_GATE_PROB_BEFORE_DROPPING(self):
        return self.moe_normalize_gate_prob_before_dropping

    @property
    def MOE_EXPERT_DROPOUT(self):
        return self.moe_expert_dropout

    @property
    def MOE_GATE_LOSS_WT(self):
        return self.moe_gate_loss_wt

    @property
    def MOE_GATE_LOSS_COMBINE_METHOD(self):
        return self.moe_gate_loss_combine_method

    @property
    def MOE_GATE_LOSS_RANK(self):
        return self.moe_gate_loss_rank

    @property
    def MOE_EOM_LOSS_WT(self):
        return self.moe_eom_loss_wt

    @property
    def ENCODER_LAYERS(self):
        return self.encoder_layers

    @property
    def ENCODER_ATTENTION_HEADS(self):
        return self.encoder_attention_heads

    @property
    def ATTENTION_DROPOUT(self):
        return self.attention_dropout

    @property
    def ACTIVATION_DROPOUT(self):
        return self.activation_dropout

    @property
    def SUBLN(self):
        return self.subln

    @property
    def XPOS_REL_POS(self):
        return self.xpos_rel_pos

    @property
    def ENCODER_NORMALIZE_BEFORE(self):
        return self.encoder_normalize_before

    @property
    def ENCODER_FFN_EMBED_DIM(self):
        return self.encoder_ffn_embed_dim

    @property
    def ACTIVATION_FN(self):
        return self.activation_fn

    @property
    def DEEPNORM(self):
        return self.deepnorm

    @property
    def FSDP(self):
        return self.fsdp

    @property
    def NORMALIZE_OUTPUT(self):
        return self.normalize_output

    @property
    def REL_POS_BUCKETS(self):
        return self.rel_pos_buckets

    @property
    def MAX_REL_POS(self):
        return self.max_rel_pos

    @property
    def BERT_INIT(self):
        return self.bert_init

    @property
    def OPT_BETAS(self):
        return self.opt_betas

    @property
    def PRETRAINED_MODEL_PATH(self):
        return self.pretrained_model_path

    @property
    def LR_BASE(self):
        return self.lr_base

    @property
    def WEIGHT_DECAY(self):
        return self.weight_decay

    @property
    def EPOCHS(self):
        return self.epochs

    @property
    def BATCH_SIZE(self):
        return self.batch_size