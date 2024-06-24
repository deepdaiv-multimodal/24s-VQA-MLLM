import torch
from torchvision.datasets.folder import default_loader
from transformers import XLMRobertaTokenizer
import yaml
from PIL import Image
from modeling_finetune import beit3_base_patch16_384_vqav2
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

class VQAnswering():
    def __init__(self, beit3_config: dict):
        self.config = beit3_config
        self.device = torch.device(self.config["device"])
        self._fix_seed()
        cudnn.benchmark = True

        # tokenizer initialize
        self.tokenizer = XLMRobertaTokenizer("/home/user/PycharmProjects/TK/beit/beit3.spm")
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.label2ans = self._label2ans(self.config["ans2label"])

        self.input_size = self.config["image_size"]
        self.transform = self._build_transform()

        # 모델 초기화 code
        self.model = beit3_base_patch16_384_vqav2(pretrained=False, **self.config)
        self.model.to(self.device)
        self.model.eval()
        print('Model initialized.')

    def predict(self, text, image_path):
        image = default_loader(image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)

        language_tokens, padding_mask, _ = self._tokenize(text)

        # 모델 예측
        with torch.no_grad():
            logits = self.model(
                image=image,
                question=language_tokens,
                padding_mask=padding_mask
            )
            probabilities = F.softmax(logits, dim=-1)
            max_prob, preds = probabilities.max(-1)
            answer = self.label2ans[preds.item()]

        return answer, max_prob.item()

    def _tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        max_len = 64
        if len(token_ids) > max_len - 2:
            token_ids = token_ids[:max_len - 2]

        token_ids = [self.bos_token_id] + token_ids[:] + [self.eos_token_id]
        num_tokens = len(token_ids)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = token_ids + [self.pad_token_id] * (max_len - num_tokens)

        language_tokens = torch.Tensor(language_tokens).type('torch.LongTensor').reshape(1, -1).to(self.device)
        padding_mask = torch.Tensor(padding_mask).type('torch.LongTensor').reshape(1, -1).to(self.device)
        return language_tokens, padding_mask, num_tokens

    def _build_transform(self):
        t = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        return t

    def _fix_seed(self, seed: int = 666666):
        torch.manual_seed(seed)

    def _label2ans(self, ans2label):
        label2ans = []
        with open(ans2label, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = yaml.safe_load(line)
                ans = data["answer"]
                label2ans.append(ans)
        return label2ans


if __name__ == "__main__":
    with open('/home/user/PycharmProjects/TK/beit/beit3/config.yml', 'r') as f:
        config = yaml.full_load(f)
    beit3_config = config["beit3"]

    vqa = VQAnswering(beit3_config)

    # 예제 질문
    text = "dogs"
    image_path = "./sample/three_puppies.jpg"

    # 예측
    result = vqa.predict(text, image_path)
    print(f"Question: {text}")
    print(f"Answer: {result}")
