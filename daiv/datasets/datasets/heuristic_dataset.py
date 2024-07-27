import os
import json
from typing import Iterable

from torch.utils.data import Dataset
from collections import Counter
from PIL import Image

from daiv.datasets.datasets.base_dataset import BaseDataset_H

def ok_score(gt_answers):
    gt_answers = [a['answer'] for a in gt_answers]
    ans2cnt = Counter(gt_answers)
    # sort
    ans2cnt = sorted(ans2cnt.items(), key=lambda x: x[1], reverse=True)
    ans2score = {}
    for ans, cnt in ans2cnt:
        # ans2score[ans] = min(1.0, cnt / 3.0)
        if cnt == 1:
            ans2score[ans] = 0.3
        elif cnt == 2:
            ans2score[ans] = 0.6
        elif cnt == 3:
            ans2score[ans] = 0.9
        else:
            ans2score[ans] = 1.0
    return ans2score

class HEURISTICDataset(BaseDataset_H):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): Directory to store the annotation file
        """
        self.vis_root = vis_root

        # annotations 
        with open(ann_paths[0], "r") as f:
            data = json.load(f)
            self.annotation = data['annotations']
        
        self.annotation_h = json.load(open(ann_paths[1]))
        
        # captions 
        self.iid_to_capt = json.load(open(ann_paths[2]))
        self.iid_to_capt_h = json.load(open(ann_paths[3]))

        # heuristic answers
        self.qid_to_topk = json.load(open(ann_paths[4]))
        self.qid_to_ex_id = json.load(open(ann_paths[5]))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # self._add_instance_ids()

        self.qid_to_ques_h = {str(q['encoded_question_id']): q for q in self.annotation_h}

        # prophet dataloder 
        qid_to_data = {}
        qid_to_ques = {str(q['question_id']): q for q in self.annotation}
        # print('qid_to_ques의 개수: ', len(qid_to_ques))

        for qid in qid_to_ques:
            q_item = qid_to_ques[qid]
            t_item = self.qid_to_topk[qid]

            # caption 
            iid = str(q_item['image_id'])
            caption = self.iid_to_capt[iid].strip()
            if caption[-1] != '.':
                caption += '.'
            
            # answer
            answers = q_item['answers']
            ans2score = ok_score(answers)
            most_answer = list(ans2score.keys())[0]
            if most_answer == '':
                most_answer = list(ans2score.keys())[1]
            # answer_texts = [answer['answer'] for answer in answers]
            # most_answer = Counter(answer_texts).most_common(1)[0][0]

            # heuristic prompt 생성
            similar_examples = self.qid_to_ex_id[qid]
            prompt_text = self.get_context(similar_examples)

            qid_to_data[qid] = {
                'question_id': qid,
                'image_id': iid,
                'question': q_item['question'],
                'most_answer': most_answer,
                # 'ans2score': ans2score,
                'topk_candidates': t_item,
                'caption': caption,
                # 'similar_qids': similar_qids
                'prompt_text': prompt_text
            }

        self.qid_to_data = qid_to_data
        self.qid_list = list(self.qid_to_data.keys())

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # data = self.qid_to_data[__key]
        qid = self.qid_list[index]
        data = self.qid_to_data[qid]
        
        # load image 
        iid = int(data['image_id'])
        image_filename = f"train2014/COCO_train2014_{iid:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_filename)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # text_input: answer heuristic
        question = self.text_processor(data['question'])
        prompt_query = self.sample_make(question, data['caption'], data['topk_candidates'])
        # prompt_in_ctx = selget_context(example_qids)
        prompt_in_ctx = data['prompt_text']
        text_input = prompt_in_ctx + prompt_query
        # text_input = self.text_processor(text_input)

        # text_output
        text_output = data['most_answer']

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            # 'weights':answer_weight
        }
    
    def sample_make(self, ques, capt, cands, ans=None):
        line_prefix = "===\n"
        prompt_text = line_prefix + f'Context: {capt}\n'
        prompt_text += line_prefix + f'Question: {ques}\n'
        cands_with_conf = [f'{cand["answer"]}({cand["confidence"]:.2f})' for cand in cands]
        cands = ', '.join(cands_with_conf)
        prompt_text += line_prefix + f'Candidates: {cands}\n'
        prompt_text += line_prefix + 'Answer:'
        if ans is not None:
            prompt_text += f' {ans}'
        return prompt_text

    def get_context(self, example_qids):
        # making context text for one testing input
        prompt_text = "Your task is to answer a knowledge-based question. Please choose the correct answer in the choices according to the context, the question, the answer candidates. Each answer candidate is associated with a confidence score within a bracket. The true answer may not be included in the candidates.\n\n"
        examples = []
        prompt_img = []
        for key in example_qids:

            key = str(key)
            data = self.qid_to_ques_h[key]

            ques = data['question']
            iid = str(data['image_id'])
            caption = self.iid_to_capt_h[iid]
            cands = self.qid_to_topk[key]
            answers = data['direct_answers']
            gt_ans = Counter(answers).most_common(1)[0][0]
            # examples.append((ques, caption, cands, gt_ans))
            p = self.sample_make(ques, caption, cands, ans=gt_ans)
            prompt_text += p
            # prompt_img.append(img)
            prompt_text += '\n\n'

        return prompt_text

class HEURISTICEvalCDataset(BaseDataset_H):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): Directory to store the annotation file
        """
        self.vis_root = vis_root

        # annotations 
        with open(ann_paths[0], "r") as f:
            data = json.load(f)
            self.annotation = data['annotations']
        
        with open(ann_paths[1], "r") as f:
            data = json.load(f)
            self.annotation_h = data['annotations']
        
        # captions 
        self.iid_to_capt = json.load(open(ann_paths[2]))
        # self.iid_to_capt_h = json.load(open(ann_paths[3]))

        # heuristic answers
        self.qid_to_topk = json.load(open(ann_paths[3]))
        self.qid_to_ex_id = json.load(open(ann_paths[4]))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # self._add_instance_ids()

        self.qid_to_ques_h = {str(q['question_id']): q for q in self.annotation_h}

        # prophet dataloder 
        qid_to_data = {}
        qid_to_ques = {str(q['question_id']): q for q in self.annotation}
        # print('qid_to_ques의 개수: ', len(qid_to_ques))

        for qid in qid_to_ques:
            q_item = qid_to_ques[qid]
            t_item = self.qid_to_topk[qid]

            # caption 
            iid = str(q_item['image_id'])
            caption = self.iid_to_capt[iid].strip()
            if caption[-1] != '.':
                caption += '.'
            
            # answer
            answers = q_item['answers']
            ans2score = ok_score(answers)
            most_answer = list(ans2score.keys())[0]
            if most_answer == '':
                most_answer = list(ans2score.keys())[1]
            # answer_texts = [answer['answer'] for answer in answers]
            # most_answer = Counter(answer_texts).most_common(1)[0][0]

            # heuristic prompt 생성
            similar_examples = self.qid_to_ex_id[qid]
            prompt_text = self.get_context(similar_examples)

            qid_to_data[qid] = {
                'question_id': qid,
                'image_id': iid,
                'question': q_item['question'],
                'most_answer': most_answer,
                # 'ans2score': ans2score,
                'topk_candidates': t_item,
                'caption': caption,
                # 'similar_qids': similar_qids
                'prompt_text': prompt_text
            }

        self.qid_to_data = qid_to_data
        self.qid_list = list(self.qid_to_data.keys())

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # data = self.qid_to_data[__key]
        qid = self.qid_list[index]
        data = self.qid_to_data[qid]
        
        # load image 
        iid = int(data['image_id'])
        image_filename = f"val2014/COCO_val2014_{iid:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_filename)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # text_input: answer heuristic
        question = self.text_processor(data['question'])
        prompt_query = self.sample_make(question, data['caption'], data['topk_candidates'])
        # prompt_in_ctx = selget_context(example_qids)
        prompt_in_ctx = data['prompt_text']
        text_input = prompt_in_ctx + prompt_query
        # text_input = self.text_processor(text_input)

        # text_output
        text_output = data['most_answer']

        # question_id
        question_id = data['question_id']

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            # 'weights':answer_weight
            "question_id": question_id
        }
    
    def sample_make(self, ques, capt, cands, ans=None):
        line_prefix = "===\n"
        prompt_text = line_prefix + f'Context: {capt}\n'
        prompt_text += line_prefix + f'Question: {ques}\n'
        cands_with_conf = [f'{cand["answer"]}({cand["confidence"]:.2f})' for cand in cands]
        cands = ', '.join(cands_with_conf)
        prompt_text += line_prefix + f'Candidates: {cands}\n'
        prompt_text += line_prefix + 'Answer:'
        if ans is not None:
            prompt_text += f' {ans}'
        return prompt_text

    def get_context(self, example_qids):
        # making context text for one testing input
        prompt_text = "Your task is to answer a knowledge-based question. Please choose the correct answer in the choices according to the context, the question, the answer candidates. Each answer candidate is associated with a confidence score within a bracket. The true answer may not be included in the candidates.\n\n"
        examples = []
        prompt_img = []
        for key in example_qids:

            key = str(key)
            data = self.qid_to_ques_h[key]

            ques = data['question']
            iid = str(data['image_id'])
            caption = self.iid_to_capt[iid]
            cands = self.qid_to_topk[key]
            answers = data['answers']
            answer_texts = [answer['answer'] for answer in answers]
            gt_ans = Counter(answer_texts).most_common(1)[0][0]
            # examples.append((ques, caption, cands, gt_ans))
            p = self.sample_make(ques, caption, cands, ans=gt_ans)
            prompt_text += p
            # prompt_img.append(img)
            prompt_text += '\n\n'

        return prompt_text