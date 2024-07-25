"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from daiv.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class ImageTextPairDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        #print(f'it {ann}')

        #self.vis_root = '/root/workspace/24s-VQA-MLLM/dataset/vg/images/VG_100K'
        #image_path = os.path.join(self.vis_root, ann["image"])
        img = ann['image_id'].replace('vg_','') + '.jpg'
        image_path = os.path.join('/root/workspace/24s-VQA-MLLM/dataset/vg/images/VG_100K',img)
        #print(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image_path = os.path.join('/root/workspace/24s-VQA-MLLM/dataset/vg/images/VG_100K_2',img)
            image = Image.open(image_path).convert("RGB")
            #return None

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {"image": image, "text_input": caption}

class ImageTextPairInstructDataset(ImageTextPairDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("Your task is to answer a knowledge-based question. Question: What caption best describes this image? Short answer:")
        return data