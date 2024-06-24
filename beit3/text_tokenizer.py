from transformers import XLMRobertaTokenizer
from datasets import VQAv2Dataset


tokenizer = XLMRobertaTokenizer("/home/user/PycharmProjects/TK/beit/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/home/user/PycharmProjects/TK/beit/data",
    tokenizer=tokenizer,
    annotation_data_path="/home/user/PycharmProjects/TK/beit/data/vqa",
)
