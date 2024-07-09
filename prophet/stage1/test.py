from utils.datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/content/drive/MyDrive/24s-deep-daiv/24s-VQA-MLLM/datasets/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/content/drive/MyDrive/24s-deep-daiv/ok-vqa",
    tokenizer=tokenizer,
    annotation_data_path="/content/drive/MyDrive/24s-deep-daiv/24s-VQA-MLLM/datasets/okvqa",
)