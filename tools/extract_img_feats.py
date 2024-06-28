# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Tool for extracting image features using BEiT-3 Encoder
# ------------------------------------------------------------------------------ #

import os, sys
sys.path.append(os.getcwd())

import glob, re, math, time, datetime
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
from transformers import BeitFeatureExtractor, BeitModel

from configs.task_cfgs import Cfgs
from configs.task_to_split import *
from tools.transforms import _transform


@torch.no_grad()
def _extract_feat(img_path, model, feature_extractor, save_path):
    img = Image.open(img_path).convert('RGB')
    inputs = feature_extractor(images=img, return_tensors="pt").to('cuda')
    outputs = model.extract_features(inputs['pixel_values'])
    beit_feats = outputs.cpu().numpy()[0]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        x=beit_feats,
    )


class ExtractModel:
    def __init__(self, encoder, feature_extractor) -> None:
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.encoder.cuda().eval()

    @torch.no_grad()
    def extract_features(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state


def main(__C, dataset):
    # find imgs
    img_dir_list = []
    for split in SPLIT_TO_IMGS:
        if split.startswith(dataset):
            img_dir_list.append(
                __C.IMAGE_DIR[SPLIT_TO_IMGS[split]]
            )
    print('image dirs:', img_dir_list)
    img_path_list = []
    for img_dir in img_dir_list:
        print(f'Checking directory: {img_dir}')  # 현재 디렉토리 출력
        img_path_list += glob.glob(os.path.join(img_dir, '*.jpg'))  # os.path.join 사용
    print('total images:', len(img_path_list))

    # load BEiT model
    feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    beit_model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

    model = ExtractModel(beit_model, feature_extractor)

    for img_path in tqdm(img_path_list):
        img_path_sep = img_path.split('/')
        img_path_sep[-3] += '_feats'
        save_path = '/'.join(img_path_sep).replace('.jpg', '.npz')
        _extract_feat(img_path, model, feature_extractor, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tool for extracting BEiT image features.')
    parser.add_argument('--dataset', dest='dataset', help='dataset name, e.g., ok, aok', type=str, required=True)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default='0')
    parser.add_argument('--clip_model', dest='CLIP_VERSION', help='clip model name or local model checkpoint path', type=str, default='RN50x64')
    parser.add_argument('--img_resolution', dest='IMG_RESOLUTION', help='image resolution', type=int, default=512)
    args = parser.parse_args()
    __C = Cfgs(args)
    main(__C, args.dataset)
