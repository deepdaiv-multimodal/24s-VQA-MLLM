"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch

from daiv.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, feature_list, question_list, prompt_list, answer_list, weight_list = [], [], [], [], [], []

        num_answers = []

        #print(f'samples : {samples}')
        
        for sample in samples:
            image_list.append(sample["image"])
            feature_list.append(sample["feats"])
            question_list.append(sample["question"])
            prompt_list.append(sample["text_input"])
            # weight_list.append(sample["weights"][sample['text_output']])
            answer_list.append(sample['text_output'])
            # num_answers.append(len(list(sample["weights"].values())))

        return {
            "image": torch.stack(image_list, dim=0),
            "feats": torch.stack(feature_list, dim=0),
            "question": torch.stack(question_list, dim=0),
            "text_input": prompt_list,
            "text_output": answer_list,
            # "weight": torch.Tensor(weight_list),
            # "n_answers": torch.LongTensor(num_answers),
        }


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    # def collater(self, samples):
    #     image_list, feature_list, question_list, prompt_list, answer_list, weight_list = [], [], [], [], [], []

    #     num_answers = []

    #     #print(f'samples : {samples}')
        
    #     for sample in samples:
    #         image_list.append(sample["image"])
    #         feature_list.append(sample["feats"])
    #         question_list.append(sample["question"])
    #         prompt_list.append(sample["text_input"])
    #         # weight_list.append(sample["weights"][sample['text_output']])
    #         answer_list.append(sample['text_output'])
    #         # num_answers.append(len(list(sample["weights"].values())))

    #     return {
    #         "image": torch.stack(image_list, dim=0),
    #         "feats": torch.stack(feature_list, dim=0),
    #         "question": torch.stack(question_list, dim=0),
    #         "text_input": prompt_list,
    #         "text_output": answer_list,
    #         # "weight": torch.Tensor(weight_list),
    #         # "n_answers": torch.LongTensor(num_answers),
    #     }
