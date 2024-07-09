import os
import time
import random
import json
import yaml
import torch
from pathlib import Path
from PIL import Image
import math
import argparse
import random

from transformers import InstructBlipProcessor
from .utils.custom import InstructBlipForConditionalGeneration
from transformers import BlipImageProcessor, AutoTokenizer, InstructBlipConfig, AutoProcessor

from .utils.fancy_pbar import progress, info_column
from .utils.data_utils import Qid2Data
from configs.task_cfgs import Cfgs

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater

        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        config = InstructBlipConfig.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.model = InstructBlipForConditionalGeneration(config).to(self.device)

    def infer_with_blip(self, image_path, prompt_text, ques, _retry=0):
        if _retry > 0:
            print('retrying...')
            st = 2 ** _retry
            time.sleep(st)

        if self.__C.DEBUG:
            time.sleep(0.05)
            return "Debug mode response", 0

        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"The image at path {image_path} does not exist.")

            image = Image.open(image_path).convert("RGB")
            # Ensure `ques` is a string or list of strings
            if not isinstance(ques, str):
              if isinstance(ques, list):
                ques = [str(q) for q in ques]
              else:
                ques = str(ques)
                
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt", truncation=True, max_length=512) #prompts=prompt_text,
            if torch.cuda.is_available():
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
            outputs, scores = self.model.generate_with_scores(
                                          **inputs,
                                          do_sample=False,
                                          num_beams=5,
                                          max_length=256,
                                          min_length=1,
                                          top_p=0.9,
                                          repetition_penalty=1.5,
                                          length_penalty=1.0,
                                          temperature=1,
                                      )

            response_txt = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            
            # print(prompt_text)
            # print(response_txt)
            
            logprobs = []
            for score in scores:
                logprobs.append(score.log_softmax(dim=-1).max(dim=-1).values)
            total_logprob = torch.sum(torch.stack(logprobs))
            prob = math.exp(total_logprob.item())
            # print(prob)

        except FileNotFoundError as e:
            print(e)
            return "Image not found", 0
        except Exception as e:
            print(type(e), e)
            return self.infer_with_blip(image_path, prompt_text, _retry + 1)

        return response_txt, prob

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
        return prompt_text, image_path

        

    def get_context(self, example_qids):
        prompt_text = self.__C.PROMPT_HEAD
        examples = []
        for key in example_qids:
            key = str(key)
            ques = self.trainset.get_question(key)
            caption = self.trainset.get_caption(key)
            cands = self.trainset.get_topk_candidates(key)
            gt_ans = self.trainset.get_most_answer(key)
            image_path = self.trainset.get_image_path(key)
            examples.append((ques, caption, cands, gt_ans, image_path))
            context_prompt, _ = self.sample_make(ques, caption, cands, image_path, ans=gt_ans)
            prompt_text += context_prompt
            prompt_text += '\n\n'
        return prompt_text

    def run(self):
        ## 로그 저장 위치 설정
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        ## 결과 저장 위치 설정
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)

        self.cache = {}
        self.cache_file_path = os.path.join(self.__C.RESULT_DIR, 'cache.json')
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))

        print('Note that the accuracies printed before final evaluation (the last printed one) are rough, just for checking if the process is normal!!!\n')
        self.trainset = Qid2Data(self.__C, self.__C.TRAIN_SPLITS, True)
        self.valset = Qid2Data(self.__C, self.__C.EVAL_SPLITS, self.__C.EVAL_NOW, json.load(open(self.__C.EXAMPLES_PATH, 'r')))

        infer_times = self.__C.T_INFER
        N_inctx = self.__C.N_EXAMPLES

        random.seed(42)
        sampled_keys = random.sample(list(self.valset.qid_to_data.keys()), 1000)
        for qid in progress.track(sampled_keys, description="Working...  "):
            if qid in self.cache:
                continue
            ques = self.valset.get_question(qid)
            caption = self.valset.get_caption(qid)
            cands = self.valset.get_topk_candidates(qid, self.__C.K_CANDIDATES)
            image_path = self.valset.get_image_path(qid)
            tags = self.valset.get_tags(qid)
            prompt_query, image_path = self.sample_make(ques, caption, cands, image_path)
            example_qids = self.valset.get_similar_qids(qid, k=infer_times * N_inctx)
            random.shuffle(example_qids)

            prompt_info_list = []
            ans_pool = {}
            # 다중 추론
            for t in range(infer_times):
                prompt_in_ctx = self.get_context(example_qids[(N_inctx * t):(N_inctx * t + N_inctx)])
                prompt_text = prompt_in_ctx + prompt_query
                gen_text, gen_prob = self.infer_with_blip(image_path, prompt_text, ques)

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

            # 다수결 투표
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
            json.dump(self.cache, open(self.cache_file_path, 'w'))

            ll = len(self.cache)
            if self.__C.EVAL_NOW and not self.__C.DEBUG:
                if ll > 21 and ll % 10 == 0:
                    rt_accuracy = self.valset.rt_evaluate(self.cache.values())
                    info_column.info = f'Acc: {rt_accuracy}'
                if 11 % 100 ==0:
                  print(rt_accuracy)

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
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH', help='candidates file path, default: "assets/candidates_for.ok.json"', type=str, default=None)
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH', help='captions file path, default: "assets/captions_for.ok.json"', type=str, default=None)
    parser.add_argument('--image_path', dest='IMAGE_PATH', help='path to the images folder', type=str, required=True)

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
