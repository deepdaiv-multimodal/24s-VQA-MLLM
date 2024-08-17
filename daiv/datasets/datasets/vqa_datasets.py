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
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers = []

        #print(f'samples : {samples}')
        '''
        samples : {
        image : tensor([[[]]])
        text_input : str (question)
        text_output : str (best_answer) .. only one
        weights : dict {cand1 : conf1 , cand2 : conf2 ...} .. 이미지마다 개수 다름 (conf합이 1)

        }
        '''

        for sample in samples:
            image_list.append(sample["feat"])
            question_list.append(sample["text_input"])
            weight_list.append(sample["weights"][sample['text_output']])
            answer_list.append(sample['text_output'])
            num_answers.append(len(list(sample["weights"].values())))

        return {
            "feat": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
