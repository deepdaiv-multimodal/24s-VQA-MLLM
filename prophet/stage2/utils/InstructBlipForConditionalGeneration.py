import torch
import torch.nn as nn
from transformers import InstructBlipPreTrainedModel, InstructBlipConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.models.instructblip.modeling_instructblip import InstructBlipVisionModel, InstructBlipQFormerModel, InstructBlipForConditionalGenerationModelOutput
from transformers import BertTokenizer, BertModel

from typing import Any, Optional, Tuple, Union, List

class PromptEncoder(nn.Module):
    def __init__(self, encoder_model, tokenizer, output_dim):
        super(PromptEncoder, self).__init__()
        self.encoder = encoder_model
        self.tokenizer = tokenizer
        self.encoder.eval()
        self.project = nn.Linear(encoder_model.config.hidden_size, output_dim)

    def forward(self, tokenized_prompts):
        with torch.no_grad():
            output = self.encoder(**tokenized_prompts)
            encoded_prompts = self.project(output.pooler_output)
        return encoded_prompts

class VisualGatedFusion(nn.Module):
    def __init__(self, text_dim, visual_dim):
        super(VisualGatedFusion, self).__init__()
        self.query_proj = nn.Linear(text_dim, text_dim)
        self.key_proj = nn.Linear(visual_dim, text_dim)
        self.value_proj = nn.Linear(visual_dim, text_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_embeddings, visual_embeddings):
        # if visual_embeddings.shape[-1] != self.key_proj.in_features:
        #     raise ValueError(f"Expected visual_embeddings with last dimension {self.key_proj.in_features}, but got {visual_embeddings.shape[-1]}")
        Q = self.query_proj(text_embeddings)
        K = self.key_proj(visual_embeddings)
        V = self.value_proj(visual_embeddings)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / K.size(-1)**0.5
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, V)
        lambda_ = self.sigmoid(context)
        Fm = (1 - lambda_) * text_embeddings + lambda_ * context
        return Fm


class ResamplerDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResamplerDecoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, fused_embeddings):
        Fp = self.linear(fused_embeddings)
        return Fp

class VisualAwarePromptingModule(nn.Module):
    def __init__(self, encoder_model, tokenizer, text_dim, visual_dim, output_dim):
        super(VisualAwarePromptingModule, self).__init__()
        self.prompt_encoder = PromptEncoder(encoder_model, tokenizer, text_dim)
        self.visual_gated_fusion = VisualGatedFusion(text_dim, visual_dim)
        self.resampler_decoder = ResamplerDecoder(text_dim, output_dim)
        self.tokenizer = tokenizer

    def forward(self, visual_embeddings, text_prompts):
        if isinstance(text_prompts, torch.Tensor):
            text_prompts = [self.tokenizer.decode(p, skip_special_tokens=True) for p in text_prompts]
        
        tokenized_prompts = self.tokenizer(text_prompts, return_tensors='pt', padding=True, truncation=True)
        tokenized_prompts = {key: value.to(visual_embeddings.device) for key, value in tokenized_prompts.items()}
        Fs = self.prompt_encoder(tokenized_prompts)
        Fm = self.visual_gated_fusion(Fs, visual_embeddings)
        Fp = self.resampler_decoder(Fm)
        return Fp


class InstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        
        self.vision_model = InstructBlipVisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        bert_model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.visual_prompting_module = VisualAwarePromptingModule(bert_model, tokenizer, config.qformer_config.hidden_size, config.vision_config.hidden_size, config.text_config.hidden_size)

        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config, attn_implementation=config._attn_implementation)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config, attn_implementation=config._attn_implementation)

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model
        self.post_init()

    def forward(self, pixel_values, qformer_input_ids, qformer_attention_mask=None, input_ids=None, attention_mask=None,
                decoder_input_ids=None, decoder_attention_mask=None, prompts=None, prompt_attention_mask=None,
                output_attentions=None, output_hidden_states=None, labels=None, return_dict=None, interpolate_pos_encoding=False):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states, return_dict=return_dict,
                                           interpolate_pos_encoding=interpolate_pos_encoding)
        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        
        query_outputs = self.qformer(input_ids=qformer_input_ids, attention_mask=qformer_attention_mask,
                                     query_embeds=query_tokens, encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=image_attention_mask, output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states, return_dict=return_dict)
        query_output = query_outputs[0][:query_tokens.size(1), :]
        
        prompted_output = self.visual_prompting_module(visual_embeddings=image_embeds, text_prompts=prompts)
        print(prompted_output.shape)
        language_model_inputs = self.language_projection(prompted_output)
        language_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                          output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                          return_dict=return_dict)
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1):, :]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                loss_fct = nn.CrossEntropyLoss(reduction="mean")
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                          decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                                          output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                          return_dict=return_dict, labels=labels)
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return InstructBlipForConditionalGenerationModelOutput(
            loss=loss, logits=logits, vision_outputs=vision_outputs, qformer_outputs=query_outputs, language_model_outputs=outputs
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        prompts: Optional[Union[str, List[str], torch.Tensor]] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        if hasattr(self, "hf_device_map"):
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        
        if qformer_input_ids is None:
            qformer_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=image_embeds.device)
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        else:
            if qformer_attention_mask is None:
                qformer_attention_mask = torch.ones_like(qformer_input_ids)
                
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

        # Ensure prompts is a list of strings if it's a tensor
        if isinstance(prompts, torch.Tensor):
            prompts = [self.visual_prompting_module.tokenizer.decode(p, skip_special_tokens=True) for p in prompts]
        
        prompted_output = self.visual_prompting_module(visual_embeddings=image_embeds, text_prompts=prompts)
        print(prompted_output.shape)
        language_model_inputs = self.language_projection(prompted_output)
        language_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        if not self.language_model.config.is_encoder_decoder:
            bos_token_id = (
                2 if self.config.text_config.architectures[0] == "LLaMAForCausalLM" else self.config.text_config.bos_token_id
            )
            bos_tokens = torch.LongTensor([[bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
            if not isinstance(outputs, torch.Tensor):
                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
            else:
                outputs = torch.cat([bos_tokens, outputs], dim=-1)

        return outputs
