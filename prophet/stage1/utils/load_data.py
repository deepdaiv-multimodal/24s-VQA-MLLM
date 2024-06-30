import numpy as np
import glob
import json
import torch
import torch.utils.data as Data
from transformers import AutoTokenizer, AutoFeatureExtractor, XLMRobertaTokenizer
from PIL import Image
from evaluation.ans_punct import prep_ans

def soft_target(answers, ans_to_ix):
    ans_score = np.zeros(len(ans_to_ix), np.float32)
    for ans in answers:
        if not isinstance(ans, str):
            ans = str(ans)
        ans = prep_ans(ans)
        if ans in ans_to_ix:
            ans_score[ans_to_ix[ans]] = min(1.0, ans_score[ans_to_ix[ans]] + 0.3)
    return ans_score

class CommonData:
    def __init__(self, __C):
        print('Loading common data...')
        
        self.imgid_to_path = {}
        for split in ['train2014', 'val2014']:
            img_dir = f"/content/drive/MyDrive/prophet/datasets/coco2014/{split}/"
            img_paths = glob.glob(img_dir + '*.jpg')
            for img_path in img_paths:
                img_id = int(img_path.split('/')[-1].split('_')[-1].split('.')[0])
                self.imgid_to_path[img_id] = img_path
        print(f'== Total image number: {len(self.imgid_to_path)}')

        self.tokenizer = XLMRobertaTokenizer("/content/drive/MyDrive/prophet/datasets/beit3.spm")
        self.token_size = self.tokenizer.vocab_size
        self.ix_to_ans = json.load(open(__C.ANSWER_DICT_PATH[__C.DATA_TAG], 'r'))
        self.ans_to_ix = {ans: ix for ix, ans in enumerate(self.ix_to_ans)}
        self.ans_size = len(self.ans_to_ix)
        print(f'== Answer vocab size: {self.ans_size}')
        print('Common data process is done.\n')

        # Cfgs 객체에 token_size 값을 설정
        __C.token_size = self.token_size

class DataSet(Data.Dataset):
    def __init__(self, __C, common_data, split_name_list):
        self.__C = __C
        self.imgid_to_path = common_data.imgid_to_path
        self.tokenizer = common_data.tokenizer
        self.ans_to_ix = common_data.ans_to_ix
        self.ix_to_ans = common_data.ix_to_ans
        self.ans_size = common_data.ans_size
        self.ques_list = []
        self.ans_list = []
        for split_name in split_name_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split_name], 'r'))['questions']
            if split_name in __C.ANSWER_PATH:
                self.ans_list += json.load(open(__C.ANSWER_PATH[split_name], 'r'))['annotations']
        self.qids = [str(ans['question_id']) for ans in self.ans_list] if len(self.ans_list) == len(self.ques_list) else [str(ques['question_id']) for ques in self.ques_list]
        self.data_size = len(self.qids)
        self.qid_to_ques = {str(ques['question_id']): ques for ques in self.ques_list}
        self.qid_to_ans = {str(ans['question_id']): ans for ans in self.ans_list}
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224-pt22k-ft22k")

    def __getitem__(self, idx):
        qid = self.qids[idx]
        ques_info = self.qid_to_ques[qid]
        ques_ids = self.tokenize_question(ques_info['question'])
        img_path = self.imgid_to_path[int(ques_info['image_id'])]
        img = self.preprocess_image(img_path)['pixel_values'][0]
        img = self.convert_to_features(img)  # 이미지 변환 추가
        ans_vec = self.soft_target(self.qid_to_ans[qid]['answers']) if self.qid_to_ans else np.zeros(self.ans_size, np.float32)
        return img, ques_ids, torch.tensor(ans_vec, dtype=torch.float)

    def __len__(self):
        return self.data_size

    def tokenize_question(self, text):
        if not isinstance(text, str):
            text = str(text)
        tokens = self.tokenizer.tokenize(text.lower().replace('?', ''))
        tokens = [self.tokenizer.cls_token] + tokens[:62] + [self.tokenizer.sep_token]
        tokens = [str(token) for token in tokens]  # ensure all tokens are strings
        return torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens + [self.tokenizer.pad_token_id] * (64 - len(tokens))), dtype=torch.long)

    def preprocess_image(self, image_path):
        return self.feature_extractor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")

    def convert_to_features(self, img):
        # 변환 함수 추가
        img = img.unsqueeze(0)
        img = nn.Conv2d(3, 4096, kernel_size=16, stride=16)(img)
        img = img.squeeze(0)
        img = img.permute(1, 2, 0).contiguous()  # (C, H, W) -> (H, W, C)
        return img

    def soft_target(self, answers):
        ans_score = np.zeros(self.ans_size, np.float32)
        for ans in answers:
            if not isinstance(ans, str):
                ans = str(ans)
            ans = prep_ans(ans)
            if ans in self.ans_to_ix:
                ans_score[self.ans_to_ix[ans]] = min(1.0, ans_score[self.ans_to_ix[ans]] + 0.3)
        return ans_score
