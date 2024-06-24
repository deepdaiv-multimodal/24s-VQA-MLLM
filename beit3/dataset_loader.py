# from datasets import VQAv2Dataset
#
# from transformers import XLMRobertaTokenizer
# data_path = "/home/user/PycharmProjects/TK/beit/data"
# split = "train"  # or "val", "test", "test-dev"
# # Initialize tokenizer (assuming using BertTokenizer here, you can use your specific tokenizer)
# tokenizer = XLMRobertaTokenizer("/home/user/PycharmProjects/TK/beit/beit3.spm")
#
# # # Define image transformations
# # transform = create_transform(
# #     input_size=480,
# #     is_training=True
# # )
#
# # Maximum number of BPE tokens
# num_max_bpe_tokens = 512  # you can set it according to your requirements
#
# # Instantiate the dataset
# vqa_dataset = VQAv2Dataset(
#     data_path=data_path,
#     split=split,
#     transform=transform,
#     tokenizer=tokenizer,
#     num_max_bpe_tokens=num_max_bpe_tokens
# )


import deepspeed
from _six import inf
from deepspeed import DeepSpeedConfig

ds_init = deepspeed.initialize
print(ds_init)