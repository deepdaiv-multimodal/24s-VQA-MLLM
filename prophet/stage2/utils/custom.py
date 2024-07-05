import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    InstructBlipPreTrainedModel,
    InstructBlipConfig,
    InstructBlipQFormerConfig,
    InstructBlipVisionConfig,
)
from transformers.utils import ModelOutput
from transformers.models.blip_2.modeling_blip_2 import Blip2VisionModel as InstructBlipVisionModel

# MCAN 모듈 가져오기
from .mcan import MCA

@dataclass
class InstructBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InstructBlipForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[torch.FloatTensor] = None
    language_model_outputs: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class InstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)

        # Vision model (encoder) to process images
        self.vision_model = InstructBlipVisionModel(config.vision_config)
        
        # MCAN model to process image embeddings and query tokens
        self.mcan = MCA(config.qformer_config.hidden_size, config.qformer_config.num_attention_heads, config.qformer_config.num_hidden_layers)

        # Projection layers to map vision and Q-Former outputs to language model's input size
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.image_embedding_projection = nn.Linear(config.vision_config.hidden_size, config.qformer_config.hidden_size)

        # Embedding layer for Q-Former input IDs
        self.qformer_input_embedding = nn.Embedding(config.qformer_config.vocab_size, config.qformer_config.hidden_size)

        # Initialize language model based on configuration
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config, attn_implementation=config._attn_implementation)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config, attn_implementation=config._attn_implementation)

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        # Initialize query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.LongTensor,  # Change to LongTensor
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through vision model to get image embeddings
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = vision_outputs[0]

        # Project image embeddings to match the input size of the Q-Former
        image_embeds_proj = self.image_embedding_projection(image_embeds)

        image_attention_mask = torch.ones(image_embeds_proj.size()[:-1], dtype=torch.long, device=image_embeds_proj.device)

        # Expand query tokens and concatenate with Q-Former input IDs
        query_tokens = self.query_tokens.expand(image_embeds_proj.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds_proj.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

        # Forward pass through MCAN

        # Ensure qformer_input_ids has the correct shape before embedding
        if qformer_input_ids.dim() == 2:
            qformer_input_ids = qformer_input_ids.unsqueeze(0)

        qformer_input_ids = self.qformer_input_embedding(qformer_input_ids)  # Use embedding layer for input IDs
        img_feat, ques_feat = self.mcan(image_embeds_proj, qformer_input_ids)
        query_output = img_feat + ques_feat

        # Project Q-Former outputs to match the input size of the language model
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        # Get embeddings for the input IDs and concatenate with Q-Former outputs and image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        # Forward pass through the language model
        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                loss_fct = CrossEntropyLoss(reduction="mean")
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return InstructBlipForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,  # Change to LongTensor
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if hasattr(self, "hf_device_map"):
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state

        # Project image embeddings to match the input size of the Q-Former
        image_embeds_proj = self.image_embedding_projection(image_embeds)

        image_attention_mask = torch.ones(image_embeds_proj.size()[:-1], dtype=torch.long, device=image_embeds_proj.device)

        # Expand query tokens and concatenate with Q-Former input IDs
        query_tokens = self.query_tokens.expand(image_embeds_proj.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds_proj.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

        # Forward pass through MCAN

        # Ensure qformer_input_ids has the correct shape before embedding
        if qformer_input_ids.dim() == 2:
            qformer_input_ids = qformer_input_ids.unsqueeze(0)

        qformer_input_ids = self.qformer_input_embedding(qformer_input_ids)  # Use embedding layer for input IDs
        img_feat, ques_feat = self.mcan(image_embeds_proj, qformer_input_ids)
        query_output = img_feat + ques_feat

        # Project Q-Former outputs to match the input size of the language model
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(language_model_inputs.device)], dim=1)

        # Get embeddings for the input IDs and concatenate with Q-Former outputs and image embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        # Adjust max_length and min_length for generation
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] + image_embeds_proj.shape[1] - 1
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1] + image_embeds_proj.shape[1]

        # Generate text using the language model
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        if not self.language_model.config.is_encoder_decoder:
            bos_token_id = (
                2 if self.config.text_config.architectures[0] == "LLaMAForCausalLM"
                else self.config.text_config.bos_token_id
            )
            bos_tokens = torch.LongTensor([[bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
            outputs = torch.cat([bos_tokens, outputs], dim=-1)

        return outputs

    def generate_with_scores(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,  # Change to LongTensor
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:

        
        outputs = self.generate(
            pixel_values=pixel_values,
            qformer_input_ids=qformer_input_ids,
            qformer_attention_mask=qformer_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **generate_kwargs,
        )

        if hasattr(outputs, "sequences") and hasattr(outputs, "scores"):
            sequences = outputs.sequences
            scores = torch.stack(outputs.scores, dim=1).squeeze(-1).mean(dim=-1)
        else:
            sequences = outputs
            scores = torch.zeros(sequences.size(0), dtype=torch.float)

        return sequences, scores
