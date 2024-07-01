import torch
import os, sys
import pickle
import json, time
import math
import random
import argparse
from datetime import datetime
from copy import deepcopy
import yaml
from pathlib import Path
import openai
import torch.multiprocessing as mp

from .utils.fancy_pbar import progress, info_column
from .utils.data_utils import Qid2Data
from configs.task_cfgs import Cfgs

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image 

def init_worker():
    torch.multiprocessing.set_start_method('spawn', force=True)

class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        # openai.api_key = __C.OPENAI_KEY

        # instructBLIP loading 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", load_in_4bit=False).to(self.device)

    def instructblip_infer(self, image_path, prompt_text, max_retries=3):
        retry_count = 0
        while retry_count < max_retries:
            try:      
                if not os.path.isfile(image_path):
                    print(f"File not found: '{image_path}', skipping...")
                    return None, None  # 이미지가 없을 경우 그냥 패스
                
                image = Image.open(image_path).convert('RGB')
                inputs = self.processor(images=image, text=prompt_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                outputs = self.model.generate(**inputs, max_length=512, return_dict_in_generate=True, output_scores=True)

                response_txt = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

                logprobs = []
                for score in outputs.scores:
                    logprobs.append(score.log_softmax(dim=-1).max(dim=-1).values)
                total_logprob = torch.sum(torch.stack(logprobs))
                prob = math.exp(total_logprob.item())

                return response_txt, prob

            except Exception as e:
                print(type(e), e)
                retry_count += 1
                if retry_count >= max_retries:
                    raise e
                time.sleep(2 ** retry_count)  # Exponential backoff

    def sample_make(self, ques, capt, cands, image_path, ans=None):
        line_prefix = self.__C.LINE_PREFIX
        cands = cands[:self.__C.K_CANDIDATES]
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
        prompt_text = self.__C.PROMPT_HEAD
        examples = []
        for key in example_qids:
            iid = int(str(key)[:-1])
            iid_path = os.path.join(self.__C.IMAGE_DIR['train2014'], f'COCO_train2014_{iid:012d}.jpg')
            ques = self.trainset.get_question(key)
            caption = self.trainset.get_caption(key)
            cands = self.trainset.get_topk_candidates(key)
            gt_ans = self.trainset.get_most_answer(key)
            examples.append((ques, caption, cands, gt_ans))
            p = self.sample_make(ques, caption, cands, iid_path, ans=gt_ans)
            prompt_text += p
            prompt_text += '\n\n'
        return prompt_text
    
    def infer_question(self, qid):
        if qid in self.cache:
            return {}
        
        ques = self.valset.get_question(qid)
        caption = self.valset.get_caption(qid)
        cands = self.valset.get_topk_candidates(qid, self.__C.K_CANDIDATES)
        image_path = self.valset.get_image_path(qid)
        image_path = str(image_path)

        prompt_query = self.sample_make(ques, caption, cands, image_path)
        example_qids = self.valset.get_similar_qids(qid, k=self.__C.T_INFER * self.__C.N_EXAMPLES)
        random.shuffle(example_qids)

        prompt_info_list = []
        ans_pool = {}
        for t in range(self.__C.T_INFER):
            prompt_in_ctx = self.get_context(example_qids[(self.__C.N_EXAMPLES * t):(self.__C.N_EXAMPLES * t + self.__C.N_EXAMPLES)])
            prompt_text = prompt_in_ctx + prompt_query
            gen_text, gen_prob = self.instructblip_infer(image_path, prompt_text)

            ans = self.evaluater.prep_ans(gen_text)
            if ans != '':
                ans_pool[ans] = ans_pool.get(ans, 0.) + gen_prob

            prompt_info = {
                'prompt': prompt_text,
                'answer': gen_text,
                'confidence': gen_prob
            }
            prompt_info_list.append(prompt_info)
            time.sleep(self.__C.SLEEP_PER_INFER)
        
        if len(ans_pool) == 0:
            answer = self.valset.get_topk_candidates(qid, 1)[0]['answer']
        else:
            answer = sorted(ans_pool.items(), key=lambda x: x[1], reverse=True)[0][0]
        
        self.evaluater.add(qid, answer)
        self.cache[qid] = {
            'question_id': qid,
            'answer': answer,
            'prompt_info': prompt_info_list
        }

        return self.cache

    def run(self):
        # num_processes = min(4, os.cpu_count())  
        num_processes = 4

        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        
        self.cache = {}
        self.cache_file_path = os.path.join(
            self.__C.RESULT_DIR,
            'cache.json'
        )
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))
        
        print('Note that the accuracies printed before final evaluation (the last printed one) are rough, just for checking if the process is normal!!!\n')
        self.trainset = Qid2Data(
            self.__C, 
            self.__C.TRAIN_SPLITS,
            True
        )

        self.valset = Qid2Data(
            self.__C, 
            self.__C.EVAL_SPLITS,
            self.__C.EVAL_NOW,
            json.load(open(self.__C.EXAMPLES_PATH, 'r'))
        )

        infer_times = self.__C.T_INFER
        N_inctx = self.__C.N_EXAMPLES

        qid_list = list(self.valset.qid_to_data.keys())

        with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
            for result in progress.track(pool.imap_unordered(self.infer_question, qid_list), total=len(qid_list), description="Working...  "):
                self.cache.update(result)
                json.dump(self.cache, open(self.cache_file_path, 'w'))

                ll = len(self.cache)
                if self.__C.EVAL_NOW and not self.__C.DEBUG:
                    if ll > 0 and ll % 10 == 0:
                        rt_accuracy = self.valset.rt_evaluate(self.cache.values())
                        info_column.info = f'Acc: {rt_accuracy}'

        self.evaluater.save(self.__C.RESULT_PATH)
        if self.__C.EVAL_NOW:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)

def prompt_login_args(parser):
    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, default='configs/prompt.yml')
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH', help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"', type=str, default=None)
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH', help='candidates file path, default: "assets/candidates_for_ok.json"', type=str, default=None)
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH', help='captions file path, default: "assets/captions_for_ok.json"', type=str, default=None)
    # parser.add_argument('--openai_key', dest='OPENAI_KEY', help='openai api key', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristics-enhanced Prompting')
    prompt_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    runner = Runner(__C, None)  

    runner.run()
