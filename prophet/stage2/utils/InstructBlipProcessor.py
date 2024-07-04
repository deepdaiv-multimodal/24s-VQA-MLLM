from transformers import BlipImageProcessor, AutoTokenizer, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

class InstructBlipProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.qformer_tokenizer = qformer_tokenizer

    def __call__(self, images=None, text=None, prompts=None, add_special_tokens=True, padding=False, truncation=None,
                 max_length=None, stride=0, pad_to_multiple_of=None, return_attention_mask=None,
                 return_overflowing_tokens=False, return_special_tokens_mask=False, return_offsets_mapping=False,
                 return_token_type_ids=False, return_length=False, verbose=True, return_tensors=None, **kwargs):

        encoding = BatchFeature()

        if text is not None:
            text_encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding,
                                           truncation=truncation, max_length=max_length, stride=stride,
                                           pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask,
                                           return_overflowing_tokens=return_overflowing_tokens,
                                           return_special_tokens_mask=return_special_tokens_mask,
                                           return_offsets_mapping=return_offsets_mapping, return_token_type_ids=return_token_type_ids,
                                           return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
            encoding.update(text_encoding)
            print(f"text encoding: {text_encoding}")

        if prompts is not None:
            prompt_encoding = self.qformer_tokenizer(text=prompts, add_special_tokens=add_special_tokens, padding=padding,
                                                     truncation=truncation, max_length=max_length, stride=stride,
                                                     pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask,
                                                     return_overflowing_tokens=return_overflowing_tokens,
                                                     return_special_tokens_mask=return_special_tokens_mask,
                                                     return_offsets_mapping=return_offsets_mapping, return_token_type_ids=return_token_type_ids,
                                                     return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
            
            encoding["prompt_input_ids"] = prompt_encoding.pop("input_ids")
            encoding["prompt_attention_mask"] = prompt_encoding.pop("attention_mask")
            print(f"prompt encoding: {prompt_encoding}")

        if images is not None:
            image_encoding = self.image_processor(images, return_tensors=return_tensors)
            encoding.update(image_encoding)
            print(f"image encoding: {image_encoding}")

        return encoding

    def batch_decode(self, *args, **kwargs):
        if 'skip_special_tokens' in kwargs:
            kwargs.pop('skip_special_tokens')
        return self.tokenizer.batch_decode(*args, skip_special_tokens=True, **kwargs)

    def decode(self, *args, **kwargs):
        if 'skip_special_tokens' in kwargs:
            kwargs.pop('skip_special_tokens')
        return self.tokenizer.decode(*args, skip_special_tokens=True, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
