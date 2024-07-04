from datasets import OKVQADataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/content/drive/MyDrive/24s-deep-daiv/24s-VQA-MLLM/datasets/beit3.spm")

OKVQADataset.make_dataset_index(
    data_path="/content/drive/MyDrive/24s-deep-daiv/ok-vqa",
    tokenizer=tokenizer,
    annotation_data_path="/content/drive/MyDrive/24s-deep-daiv/ok-vqa",
)

# from datasets import create_downstream_dataset
# data_loader_train, data_loader_val = create_downstream_dataset(is_eval=False)
# print(data_loader_train)
