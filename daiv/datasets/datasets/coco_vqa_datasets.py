"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image

from daiv.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict
import torch
import numpy as np

from daiv.datasets.data_utils import tokenize, proc_ques

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answers"]),
                "image": sample["image"],
            }
        )

class COCOVQADataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
        self.prompts = [
            "{}",
            "Question: {} Short answer:",
            "Your task is to answer a knowledge-based question. Question: {} Short answer:",
            "Using your knowledge, answer the following question: {}",
            "Given the image and your knowledge, answer the following question with no more than three words. {}",
            "Based on the image and your knowledge, respond to this question with a short answer: {}. Answer:",
            "Use the provided image and your knowledge to answer the question: {} Provide your answer as short as possible:",
            "What is the knowledge-based answer to the following question? {}",
            "The question {} can be answered using the image and your knowledge. A short answer is",
        ]

        # Tokenize
        self.stat_ques_list = []
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/workspace/24s-VQA-MLLM/dataset/vqav2/v2_OpenEnded_mscoco_test2015_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/okvqa/OpenEnded_mscoco_train2014_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/okvqa/OpenEnded_mscoco_val2014_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_train.json', 'r'))
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_val.json', 'r'))
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_test.json', 'r'))
        
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, use_glove=True)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_filename = f"train2014/COCO_train2014_{ann['image_id']:012d}.jpg"
        # image_path = os.path.join(self.vis_root, image_filename)

        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)
        feat_filename = f"train2014/{ann['image_id']}.npz"
        feat_path = os.path.join(self.vis_root, feat_filename)
        feats = np.load(feat_path)
        feats = feats['x']#.transpose((1,0))

        question = self.text_processor(ann["question"])
        ques_ix_iter = proc_ques(ann["question"], self.token_to_ix, max_token=14)

        choice = np.random.choice(len(self.prompts))
        text_input = self.prompts[choice].format(question)

        answer_weight = {}
        for answer in ann["answers"]:
            if answer["answer"] in answer_weight.keys():
                answer_weight[answer["answer"]] += 1 / len(ann["answers"])
            else:
                answer_weight[answer["answer"]] = 1 / len(ann["answers"])

        best_answer = max(answer_weight, key=answer_weight.get)

        return {
            # "image": image,
            "feats": feats,
            "question": ques_ix_iter,
            "text_input": text_input,
            "text_output": best_answer,
            'weights': answer_weight,
            'pretrained_emb': self.pretrained_emb
        }

class VQGCOCOVQADataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
        self.prompts =  ["Given the image, generate a question whose answer is: {}. Question:",
            "Based on the image, provide a question with the answer: {}. Question:",
             "Given the visual representation, create a question for which the answer is {}.",
             "From the image provided, craft a question that leads to the reply: {}. Question:",
             "Considering the picture, come up with a question where the answer is: {}.",
             "Taking the image into account, generate an question that has the answer: {}. Question:"]
        
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        choice = np.random.choice(len(self.prompts))

        text_input = self.prompts[choice].format(question)
        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        # answers = list(answer_weight.keys())
        # weights = list(answer_weight.values())
        best_answer = max(answer_weight, key=answer_weight.get)

        return {
            "image": image,
            "text_input": best_answer, #text_input,
            "text_output": text_input ,  #best_answer,
        }
    
    def collater(self, samples):
        image_list, question_list, answer_list = [], [], [],

        for sample in samples:
            image_list.append(sample["image"])
           
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.append(answers)
        

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }
        
class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): Directory to store the annotation file
        """
        self.vis_root = vis_root

        with open(ann_paths[0], "r") as f:
            data = json.load(f)
            self.annotation = data['annotations']  

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        # Tokenize
        self.stat_ques_list = []
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/workspace/24s-VQA-MLLM/dataset/vqav2/v2_OpenEnded_mscoco_test2015_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/okvqa/OpenEnded_mscoco_train2014_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/okvqa/OpenEnded_mscoco_val2014_questions.json', 'r'))['questions']
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_train.json', 'r'))
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_val.json', 'r'))
        self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_test.json', 'r'))

        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, use_glove=True)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_filename = f"val2014/COCO_val2014_{ann['image_id']:012d}.jpg"
        # image_path = os.path.join(self.vis_root, image_filename)
        # if not os.path.exists(image_path):
        # #    # 이미지가 없으면 다음 항목으로 넘어갑니다.
        # #    print(f"Warning: File {image_path} does not exist in . Skipping this item.")
        #     return self.__getitem__((index + 1) % len(self))
        # image = Image.open(image_path).convert("RGB")

        feat_filename = f"val2014/{ann['image_id']}.npz"
        feat_path = os.path.join(self.vis_root, feat_filename)
        feats = np.load(feat_path)
        feats = feats['x']

        # image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        ques_ix_iter = proc_ques(question, self.token_to_ix, max_token=14)

        return {
            # "image": image,
            "feats": feats,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

