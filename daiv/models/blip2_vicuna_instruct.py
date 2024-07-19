import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers

from daiv.models.dmformer.mcan.net import Net  # Importing the Net class from net.py
from daiv.models.dmformer.mcan.net_utils import LayerNorm  # Importing LayerNorm

from daiv.common.registry import registry
from daiv.models.blip2 import Blip2Base, disabled_train

@registry.register_model("blip2_vicuna_instruct")
class Blip2VicunaInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from daiv.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2_vicuna7b.yaml",
    }

    class Config:
        HIDDEN_SIZE = 512
        DROPOUT_R = 0.1
        MULTI_HEAD = 8
        HIDDEN_SIZE_HEAD = HIDDEN_SIZE // MULTI_HEAD
        FF_SIZE = 2048
        LAYER = 6
        FLAT_MLP_SIZE = 512
        FLAT_GLIMPSES = 1
        FLAT_OUT_SIZE = 512
        WORD_EMBED_SIZE = 300
        USE_GLOVE = False
        IMG_FEAT_SIZE = 1408  # This should match the output feature size of the visual encoder

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        llm_model="lmsys/vicuna-7b-v1.5",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # Initialize MCAN
        self.MCAN = Net(self.Config, pretrained_emb=None, token_size=len(self.tokenizer), answer_size=self.Config.HIDDEN_SIZE)

        # Initialize LLM Vicuna
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False)
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.llm_tokenizer(
            self.llm_tokenizer.eos_token, add_special_tokens=False
        ).input_ids[0]

        self.llm_proj = nn.Linear(
            self.Config.HIDDEN_SIZE, self.llm_model.config.hidden_size
        )

        self.text_proj = nn.Linear(
            self.Config.WORD_EMBED_SIZE, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_features = self.visual_encoder.get_intermediate_layers(image)[-2]  # Get image features
            image_features = image_features[:, 1:]  # Remove CLS token
            image_embeds = self.llm_proj(image_features)  # Project to LLM dimension
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text_for_mcan = samples["text_input"]
        text_for_llm = samples["text_output"]

        # Process text for MCAN
        text_tokens_mcan = self.tokenizer(
            text_for_mcan, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
        ).input_ids.to(image.device)
        text_embeds_mcan = self.MCAN.embedding(text_tokens_mcan)
        text_embeds_mcan, _ = self.MCAN.lstm(text_embeds_mcan)
        text_atts_mcan = torch.ones(text_embeds_mcan.size()[:-1], dtype=torch.long).to(image.device)

        # Process text for LLM
        text_tokens_llm = self.tokenizer(
            text_for_llm, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
        ).input_ids.to(image.device)
        text_embeds_llm = self.text_proj(self.MCAN.embedding(text_tokens_llm))
        text_atts_llm = torch.ones(text_embeds_llm.size()[:-1], dtype=torch.long).to(image.device)

        mcan_output = self.MCAN.backbone(text_embeds_mcan, image_embeds, text_atts_mcan, image_atts)

        inputs_llm = torch.cat([self.llm_proj(mcan_output), text_embeds_llm], dim=1)
        atts_llm = torch.cat([torch.ones(mcan_output.size()[:-1], dtype=torch.long).to(image.device), text_atts_llm], dim=1)

        self.llm_tokenizer.padding_side = "right"

        text = [t + self.llm_tokenizer.eos_token for t in samples["text_output"]]

        llm_tokens = self.llm_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = llm_tokens.input_ids.masked_fill(
            llm_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.model.embed_tokens(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_features = self.visual_encoder.get_intermediate_layers(image)[-2]  # Get image features
            image_features = image_features[:, 1:]  # Remove CLS token
            image_embeds = self.llm_proj(image_features)  # Project to LLM dimension
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text_for_mcan = samples["text_input"]
            text_for_llm = samples["text_output"]

            # Process text for MCAN
            text_tokens_mcan = self.tokenizer(
                text_for_mcan, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
            ).input.ids.to(image.device)
            text_embeds_mcan = self.MCAN.embedding(text_tokens_mcan)
            text_embeds_mcan, _ = self.MCAN.lstm(text_embeds_mcan)
            text_atts_mcan = torch.ones(text_embeds_mcan.size()[:-1], dtype=torch.long).to(image.device)

            # Process text for LLM
            text_tokens_llm = self.tokenizer(
                text_for_llm, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
            ).input.ids.to(image.device)
            text_embeds_llm = self.text_proj(self.MCAN.embedding(text_tokens_llm))
            text_atts_llm = torch.ones(text_embeds_llm.size()[:-1], dtype=torch.long).to(image.device)

            mcan_output = self.MCAN.backbone(text_embeds_mcan, image_embeds, text_atts_mcan, image_atts)

            inputs_llm = torch.cat([self.llm_proj(mcan_output), text_embeds_llm], dim=1)
            atts_llm = torch.cat([torch.ones(mcan_output.size()[:-1], dtype=torch.long).to(image.device), text_atts_llm], dim=1)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            llm_tokens = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            inputs_embeds = self.llm_model.model.embed_tokens(llm_tokens.input.ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.llm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            output_text = [text.strip() for text in output_text]
            return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_features = self.visual_encoder.get_intermediate_layers(image)[-2]  # Get image features
            image_features = image_features[:, 1:]  # Remove CLS token
            image_embeds = self.llm_proj(image_features)  # Project to LLM dimension
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text_for_mcan = samples["text_input"]
            text_for_llm = samples["text_output"]

            # Process text for MCAN
            text_tokens_mcan = self.tokenizer(
                text_for_mcan, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
            ).input.ids.to(image.device)
            text_embeds_mcan = self.MCAN.embedding(text_tokens_mcan)
            text_embeds_mcan, _ = self.MCAN.lstm(text_embeds_mcan)
            text_atts_mcan = torch.ones(text_embeds_mcan.size()[:-1], dtype=torch.long).to(image.device)

            # Process text for LLM
            text_tokens_llm = self.tokenizer(
                text_for_llm, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
            ).input.ids.to(image.device)
            text_embeds_llm = self.text_proj(self.MCAN.embedding(text_tokens_llm))
            text_atts_llm = torch.ones(text_embeds_llm.size()[:-1], dtype=torch.long).to(image.device)

            mcan_output = self.MCAN.backbone(text_embeds_mcan, image_embeds, text_atts_mcan, image_atts)

            inputs_llm = torch.cat([self.llm_proj(mcan_output), text_embeds_llm], dim=1)
            atts_llm = torch.cat([torch.ones(mcan_output.size()[:-1], dtype=torch.long).to(image.device), text_atts_llm], dim=1)

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            inputs_embeds = self.llm_model.model.embed_tokens(llm_tokens.input.ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.llm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llm_model = cfg.get("llm_model", "lmsys/vicuna-7b-v1.5")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
