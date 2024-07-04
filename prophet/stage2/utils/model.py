import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union
from .registry import registry
from .blip2 import Blip2Base, disabled_train

class InstructBlipForConditionalGenerationModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    vision_outputs: Optional[Tuple[torch.FloatTensor]] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

@registry.register_model("flant5xxl")
class InstructBlipForConditionalGeneration(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xxl": "/root/workspace/EunJuPark/24s-VQA-MLLM/outputs/model/daiv_flant5xxl.yaml",
    }

    def __init__(self, config):
        Blip2Base.__init__(config)

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.vision_model, self.ln_vision = self.init_vision_encoder(
            config.vision_config, config.img_size, config.drop_path_rate, config.use_grad_checkpoint, config.vit_precision
        )

        if config.freeze_vit:
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = False
            self.vision_model = self.vision_model.eval()
            self.vision_model.train = disabled_train
            logging.info("freeze vision encoder")

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer, self.query_tokens = self.init_Qformer(config.num_query_tokens, self.vision_model.num_features)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        if config.use_decoder_only_language_model:
            self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            self.language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        for name, param in self.language_model.named_parameters():
            param.requires_grad = False

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(config.text_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(config.text_model, truncation_side='right')

        self.t5_proj = nn.Linear(config.qformer_config.hidden_size, self.language_model.config.hidden_size)
        self.max_txt_len = config.max_txt_len
        self.max_output_txt_len = config.max_output_txt_len
        self.prompt = config.prompt

    def init_Qformer(self, num_query_token, vision_num_features):
        # Q-former 초기화
        return nn.Module(), nn.Parameter(torch.zeros(1, num_query_token, vision_num_features))

    def forward(self, samples):
        image = samples["image"]
        image_features = self.vision_model.get_intermediate_layers(image)[-2]
        image_features = image_features[:, 1:]
        add_feature_llm = self.vision_project(image_features)
        atts_add_feature_llm = torch.ones(add_feature_llm.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.vision_model(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        query_output = self.qformer(
            input_ids=text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            samples["text_input"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        output_tokens = self.t5_output_tokenizer(
            samples["text_output"],
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            return_tensors="pt",
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, atts_add_feature_llm, input_tokens.attention_mask], dim=1)
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        inputs_embeds = self.language_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, add_feature_llm, inputs_embeds], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(self, samples, use_nucleus_sampling=False, num_beams=5, max_length=256, min_length=1, top_p=0.9, 
                 repetition_penalty=1.5, length_penalty=1.0, num_captions=1, temperature=1):

        prompt = samples.get("prompt", self.prompt)
        image = samples["image"]
        bs = image.size(0)

        prompt = [prompt] * bs if isinstance(prompt, str) else prompt

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        text_Qformer = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        ).to(image.device)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        image_embeds = self.ln_vision(self.vision_model(image))
        image_features = self.vision_model.get_intermediate_layers(image)[-2]
        image_features = image_features[:, 1:]
        add_feature_llm = self.vision_project(image_features)
        atts_add_feature_llm = torch.ones(add_feature_llm.size()[:-1], dtype=torch.long).to(image.device)

        query_output = self.qformer(
            input_ids=text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device),
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(prompt, padding="longest", return_tensors="pt").to(image.device)
        encoder_atts = torch.cat([atts_t5, atts_add_feature_llm, input_tokens.attention_mask], dim=1)

        inputs_embeds = self.language_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, add_feature_llm, inputs_embeds], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text
