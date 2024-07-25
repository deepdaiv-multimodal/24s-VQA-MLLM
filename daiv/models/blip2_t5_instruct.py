import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

#from transformers import T5TokenizerFast, T5ForConditionalGeneration, T5Config
from daiv.models.modeling_t5 import T5Config, T5ForConditionalGeneration
import transformers
from transformers import T5TokenizerFast

from daiv.models.dmformer.mcan.net import Net  # Importing the Net class from net.py
from daiv.models.dmformer.mcan.net_utils import LayerNorm  # Importing LayerNorm

from daiv.common.registry import registry
from daiv.models.blip2 import Blip2Base, disabled_train

@registry.register_model("blip2_t5_instruct")
class Blip2T5Instruct(Blip2Base):
    """
    BLIP2 T5 model with MCAN integration.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2_instruct_flant5xl.yaml",
        "flant5xxl": "configs/models/blip2_instruct_flant5xxl.yaml",
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
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")

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

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            if param.dtype == torch.float32:  # Ensure parameters are casted correctly
                param.data = param.data.bfloat16()

        self.vision_proj = nn.Linear(self.Config.IMG_FEAT_SIZE, self.t5_model.config.hidden_size)
        self.t5_proj = nn.Linear(self.Config.HIDDEN_SIZE, self.t5_model.config.hidden_size)
        self.text_embed_proj = nn.Linear(self.Config.WORD_EMBED_SIZE, self.t5_model.config.hidden_size)

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            # # image linear projection 
            # image_features = self.visual_encoder.get_intermediate_layers(image)[-2]  # Get image features from the second to last layer
            # image_features = image_features[:, 1:]  # Remove CLS token
            # image_embeds_llm = self.vision_proj(image_features)  # Project to LLM dimension
            # image_atts_llm = torch.ones(image_embeds_llm.size()[:-1], dtype=torch.long).to(image.device)

            # Generate image_embeds_mcan as in stage1
            image_embeds_mcan = self.ln_vision(self.visual_encoder(image))
            image_embeds_mcan = self.MCAN.img_feat_linear(image_embeds_mcan)  # Project to MCAN dimension
            image_atts_mcan = self.MCAN.make_mask(image_embeds_mcan).to(image.device)

        text_input_mcan = samples["text_input"]

        # Process text for MCAN
        text_tokens_mcan = self.tokenizer(
            text_input_mcan, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
        ).input_ids.to(image.device)
        text_embeds_mcan = self.MCAN.embedding(text_tokens_mcan)
        text_embeds_mcan, _ = self.MCAN.lstm(text_embeds_mcan)
        text_atts_mcan = self.MCAN.make_mask(text_tokens_mcan.unsqueeze(2))

        txt_mcan_output, img_mcan_output = self.MCAN.backbone(text_embeds_mcan, image_embeds_mcan, text_atts_mcan, image_atts_mcan) # torch.Size([4, 257, 512])
        img_mcan_output = self.MCAN.attflat_img(img_mcan_output, image_atts_mcan) # torch.Size([4, 512])
        txt_mcan_output = self.MCAN.attflat_lang(txt_mcan_output, text_atts_mcan)

        inputs_t5 = self.t5_proj(img_mcan_output) # torch.Size([4, 2048])
        inputs_t5 = inputs_t5.unsqueeze(1) # torch.Size([4, 1, 2048])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device) # torch.Size([4, 1])

        # fs_embeds, fs_atts = None, None
        # if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
        #     fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples['few_shot_samples'])

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device) # torch.Size([4, 22])

            output_tokens = self.t5_output_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device) # torch.Size([4, 20])

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1) # torch.Size([4, 23])

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            ) # torch.Size([4, 20])

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids) # torch.Size([4, 22, 2048])
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1) # torch.Size([4, 23, 2048])

            # if fs_embeds is not None:
            #     inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
            #     encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
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
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_features = self.visual_encoder.get_intermediate_layers(image)[-2]  # Get image features from the second to last layer
            image_features = image_features[:, 1:]  # Remove CLS token
            image_embeds_llm = self.vision_proj(image_features)  # Project to LLM dimension
            image_atts_llm = torch.ones(image_embeds_llm.size()[:-1], dtype=torch.long).to(image.device)

            # Generate image_embeds_mcan as in stage1
            image_embeds_mcan = self.ln_vision(self.visual_encoder(image))
            image_embeds_mcan = self.MCAN.img_feat_linear(image_embeds_mcan)  # Project to MCAN dimension
            image_atts_mcan = self.MCAN.make_mask(image_embeds_mcan).to(image.device)

        text_input_mcan = samples["text_input"]
        text_input_llm = samples["text_input"]

        # Process text for MCAN
        text_tokens_mcan = self.tokenizer(
            text_input_mcan, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
        ).input_ids.to(image.device)
        text_embeds_mcan = self.MCAN.embedding(text_tokens_mcan)
        text_embeds_mcan, _ = self.MCAN.lstm(text_embeds_mcan)
        text_atts_mcan = self.MCAN.make_mask(text_tokens_mcan.unsqueeze(2))

        txt_mcan_output, img_mcan_output = self.MCAN.backbone(text_embeds_mcan, image_embeds_mcan, text_atts_mcan, image_atts_mcan)

        img_mcan_output = self.MCAN.attflat_img(img_mcan_output, image_atts_mcan)
        txt_mcan_output = self.MCAN.attflat_lang(txt_mcan_output, text_atts_mcan)
        
        mcan_output = img_mcan_output + txt_mcan_output
        mcan_output = self.MCAN.proj_norm(mcan_output)
        mcan_output = torch.sigmoid(self.MCAN.proj(mcan_output))

        combined_features_mcan = mcan_output.unsqueeze(1)

        # combined_features_mcan = torch.cat([img_mcan_output, txt_mcan_output], dim=1)

        text_embeds_llm_mcan = self.text_proj(combined_features_mcan)
        atts_llm_mcan = torch.ones(text_embeds_llm_mcan.size()[:-1], dtype=torch.long).to(image.device)

        # Process text for LLM
        text_tokens_llm = self.tokenizer(
            text_input_llm, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len
        ).input_ids.to(image.device)
        text_embeds_llm = self.text_embed_proj(self.MCAN.embedding(text_tokens_llm))
        atts_llm_text = torch.ones(text_embeds_llm.size()[:-1], dtype=torch.long).to(image.device)

        inputs_llm = torch.cat([image_embeds_llm, text_embeds_llm_mcan, text_embeds_llm], dim=1)
        atts_llm = torch.cat([image_atts_llm, atts_llm_mcan, atts_llm_text], dim=1)
        # inputs_llm = torch.cat([image_embeds_llm, text_embeds_llm_mcan], dim=1)
        # atts_llm = torch.cat([image_atts_llm, atts_llm_mcan], dim=1)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            # prompt = self.prompt
            prompt = samples["text_input"]

        # prompt = [prompt] * image.size(0)
        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(0), "The number of prompts must be equal to the batch size."

        t5_tokens = self.t5_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)
        attention_mask = torch.cat([atts_llm, t5_tokens.attention_mask], dim=1)

        inputs_embeds = self.t5_model.encoder.embed_tokens(t5_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

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
        length_penalty=-1,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

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
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
        )

        model.load_checkpoint_from_config(cfg)

        return model
