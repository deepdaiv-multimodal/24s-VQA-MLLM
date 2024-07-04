from transformers import XLMRobertaTokenizer
import os
import json
import random
import torch
import glob
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform

import utils
from .glossary import normalize_word
from .randaug import RandomAugment

input_size=224

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, split, transform, 
        tokenizer, num_max_bpe_tokens, task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        self.image_path = '/content/drive/MyDrive/24s-deep-daiv/ok-vqa'
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.image_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


class OKVQADataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        ans2label_file = os.path.join(data_path, "/content/drive/MyDrive/24s-deep-daiv/ok-vqa/answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                # ans = data["answer"]
                # label = data["label"]
                # label = int(label)
                # assert label == i
                # ans2label[ans] = i
                # label2ans.append(ans)
        
        # self.ans2label = ans2label
        # self.label2ans = label2ans
        self.answer = data

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("/content/drive/MyDrive/24s-deep-daiv/ok-vqa/okvqa.train.jsonl", "/content/drive/MyDrive/24s-deep-daiv/ok-vqa/okvqa.trainable_val.jsonl")
        elif split == "val":
            return ("/content/drive/MyDrive/24s-deep-daiv/ok-vqa/okvqa.rest_val.jsonl", )
        # elif split == "test":
        #     return ("vqa.test.jsonl", )
        # elif split == "test-dev":
        #     return ("vqa.test-dev.jsonl", )            
        # else:
        #     raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        # if "labels" in self.items[index] and len(self.items[index]["labels"]) > 0:
        #     labels = [0.] * len(self.label2ans)
        #     for l, s in zip(self.items[index]["labels"], self.items[index]["scores"]):
        #         labels[l] = s
        #     data["labels"] = torch.FloatTensor(labels)
        # else:
        data["qid"] = self.items[index]["qid"]
        return data

    @staticmethod
    def get_score(occurrences):
        if occurrences == 0:
            return 0.0
        elif occurrences == 1:
            return 0.3
        elif occurrences == 2:
            return 0.6
        elif occurrences == 3:
            return 0.9
        else:
            return 1.0

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, annotation_data_path):
        print("Working")
        
        print(f"annotation path: {annotation_data_path}")
        print(f"data path: {data_path}")
        
        # Load questions
        with open(os.path.join(annotation_data_path, "OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]

        print(f"Loaded {len(questions_train2014)} training questions.")
        print(f"Loaded {len(questions_val2014)} validation questions.")

        # Load annotations
        with open(os.path.join(annotation_data_path, "mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(annotation_data_path, "mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]

        print(f"Loaded {len(annotations_train2014)} training annotations.")
        print(f"Loaded {len(annotations_val2014)} validation annotations.")

        annotations = dict()

        # Process questions
        for split, questions in zip(
            ["train", "val"],
            [questions_train2014, questions_val2014],
        ):
            _annot = defaultdict(dict)
            for q in questions:
                question_text = q["question"]
                tokens = tokenizer.tokenize(question_text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                # assert q["question_id"] not in _annot[q["image_id"]]
                _annot[q["image_id"]][q["question_id"]] = {
                    "question": question_text, 
                    "token_ids": token_ids, 
                }

            annotations[split] = _annot

        all_major_answers = list()

        # Process annotations
        for split, annots in zip(
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            for q in annots:
                for answer in q["answers"]:
                    all_major_answers.append(answer["answer"])

        all_major_answers = [word.lower() for word in all_major_answers]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())

        for split, annots in zip(
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            _annot = annotations[split]
            for q in annots:
                answers = q["answers"]
                answer_count = {}
                for answer in answers:
                    answer_ = answer["answer"].lower()
                    answer_count[answer_] = answer_count.get(answer_, 0) + 1

                labels = []
                scores = []
                for answer in answer_count:
                    if answer not in ans2label:
                        continue
                    labels.append(ans2label[answer])
                    score = cls.get_score(answer_count[answer])
                    scores.append(score)

                # assert "labels" not in _annot[q["image_id"]][q["question_id"]]
                # assert "question" in _annot[q["image_id"]][q["question_id"]]
                _annot[q["image_id"]][q["question_id"]]["labels"] = labels
                _annot[q["image_id"]][q["question_id"]]["scores"] = scores

        # for split in ["train", "val"]:
        #     filtered_annot = dict()
        #     for ik, iv in annotations[split].items():
        #         new_q = dict()
        #         for qk, qv in iv.items():
        #             if len(qv["labels"]) != 0:
        #                 new_q[qk] = qv
        #         if len(new_q) != 0:
        #             filtered_annot[ik] = new_q
        #     annotations[split] = filtered_annot
        # print(annotations['train'])
        # exit()

        split2items = {}
        missing_ids = {"train": [], "val": []}
        for split in ["train", "val"]:
            annot = annotations[split]
            split_name = {
                "train": "train2014_vqa",
                "val": "val2014_vqa",
            }[split]
            paths = list(glob.glob(f"{data_path}/{split_name}/*.jpg"))
            print(f"Found {len(paths)} image paths in {split_name}.")
            random.shuffle(paths)
            annot_paths = [path for path in paths if int(os.path.basename(path).split("_")[-1].split(".")[0]) in annot]

            print(f"Found {len(annot_paths)} matching image paths in {split_name} with annotations.")
            
            items = []
            missing_paths = []
            for path in paths:
                iid = int(os.path.basename(path).split("_")[-1].split(".")[0])
                if iid not in annot:
                    missing_ids[split].append(iid)
                    missing_paths.append(path)
                    continue
                _annot = annotations[split][iid]
                for qid in _annot:
                    q = _annot[qid]
                    labels = q["labels"]
                    scores = q["scores"]

                    items.append({
                        "image_path": os.path.join(split_name, os.path.basename(path)), 
                        "text_segment": q["token_ids"], 
                        "labels": labels, 
                        "scores": scores, 
                        "qid": qid, 
                    })
            if missing_paths:
                print(f"Missing paths in {split_name}: {len(missing_paths)}")
                for i, path in enumerate(missing_paths[:10]):
                    iid = int(os.path.basename(path).split("_")[-1].split(".")[0])
                    print(f"Missing path {i+1}: {path}, Image ID: {iid}")
            split2items[split] = items

            _write_data_into_jsonl(items=items, jsonl_file=os.path.join(data_path, f"okvqa.{split}.jsonl"))

        val_image2items = defaultdict(list)
        for item in split2items["val"]:
            val_image2items[item["image_path"]].append(item)
        
        print("Contains %d images and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

        val_images = list(val_image2items.keys())
        random.shuffle(val_images)
        trainable_val = []
        rest_val = []
        for i, image_id in enumerate(val_images):
            if i < 1000:
                rest_val += val_image2items[image_id]
            else:
                trainable_val += val_image2items[image_id]
        
        _write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(data_path, "okvqa.trainable_val.jsonl"))
        _write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(data_path, "okvqa.rest_val.jsonl"))

        with open(os.path.join(data_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans, 
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))

        # Print missing IDs
        for split, ids in missing_ids.items():
            if ids:
                print(f"Missing IDs in {split}: {len(ids)}")
                for i, iid in enumerate(ids[:10]):
                    print(f"Missing ID {i+1}: {iid}")

def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))

def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train):
    # if args.task in ["imagenet"]:
    #     return build_imagenet_transform(is_train, args)

    if is_train:
        t = [
            RandomResizedCropAndInterpolation(input_size, scale=(0.5, 1.0), interpolation='bicubic'), 
            transforms.RandomHorizontalFlip(),
        ]
        # if args.randaug:
        #     t.append(
        #         RandomAugment(
        #             2, 7, isPIL=True, 
        #             augs=[
        #                 'Identity','AutoContrast','Equalize','Brightness','Sharpness', 
        #                 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
        #             ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def get_sentencepiece_model_for_beit3():
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer('/content/drive/MyDrive/24s-deep-daiv/24s-VQA-MLLM/datasets/beit3.spm')


def create_dataset_by_split(split, is_train=True):
    transform = build_transform(is_train=is_train)
    # dataset_class = task2dataset["vqav2"]
    # dataset_class = OKVQADataset()
    tokenizer = get_sentencepiece_model_for_beit3()

    opt_kwargs = {}
    # if args.task in ["coco_captioning", "nocaps"]:
    #     opt_kwargs["mask_prob"] = args.captioning_mask_prob

    dataset = OKVQADataset(
        data_path='/content/drive/MyDrive/24s-deep-daiv/24s-VQA-MLLM', split=split, 
        transform=transform, tokenizer=tokenizer, 
        num_max_bpe_tokens=64, 
        task='okvqa', **opt_kwargs, 
    )

    if is_train:
        batch_size = 16
    # elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
    #     batch_size = args.eval_batch_size
    # else:
    #     batch_size = int(args.batch_size * 1.5)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=4, pin_mem=True, dist_eval=False, 
    )


def create_downstream_dataset(is_eval=False):
    if is_eval:
        return create_dataset_by_split(split="test", is_train=False)
    else:
        return \
            create_dataset_by_split(split="train", is_train=True), \
            create_dataset_by_split(split="val", is_train=True)

# if __name__ == "__main__":
#   tokenizer = XLMRobertaTokenizer("/content/drive/MyDrive/24s-deep-daiv/24s-VQA-MLLM/datasets/beit3.spm")

#   OKVQADataset.make_dataset_index(
#       data_path="/content/drive/MyDrive/24s-deep-daiv/ok-vqa",
#       tokenizer=tokenizer,
#       annotation_data_path="/content/drive/MyDrive/24s-deep-daiv/ok-vqa",
#   )