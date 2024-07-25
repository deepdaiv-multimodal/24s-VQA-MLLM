import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from daiv.common.registry import registry
from daiv.models.base_model import all_gather_with_grad, concat_all_gather
from daiv.models.blip2 import Blip2Base, compute_sim_matrix, disabled_train
from daiv.models.blip_outputs import BlipOutput, BlipOutputFeatures

from daiv.models.dmformer.mcan.net import Net  # Importing the Net class from net.py
from daiv.models.dmformer.mcan.net_utils import LayerNorm  # Importing LayerNorm
from daiv.models.dmformer.dat.deformable_attention_1d import DeformableAttention1D

@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")

class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with MCAN and ViT.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
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
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=512,
        max_txt_len=100,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        # Initialize the vision encoder
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # Initialize MCAN instead of Q-former
        self.MCAN = Net(self.Config, pretrained_emb=None, token_size=len(self.tokenizer), answer_size=embed_dim)  # Adjust arguments as necessary
        
        #self.MCAN.resize_token_embeddings(len(self.tokenizer))


        #self.vision_proj = nn.Linear(self.Config.IMG_FEAT_SIZE, embed_dim)
        #self.text_proj = nn.Linear(self.Config.WORD_EMBED_SIZE, self.Config.HIDDEN_SIZE)  # Adjusted to match the embedding size

        self.itm_head = nn.Linear(embed_dim, 2)

        # Assuming we have a language modeling head
        self.lm_head = nn.Linear(embed_dim, len(self.tokenizer))


        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        ##DAT ATTN
        self.dat = DeformableAttention1D(
                            dim = 257,
                            downsample_factor = 4,
                            offset_scale = 2,
                            offset_kernel_size = 6,
                            offset_groups = 1
                        )

    def forward(self, samples):
        print('MCAN training....')
        image = samples["image"]
        text = samples["text_input"]

        #print(f'image size :{image.size()}')
        #print(f'text size : {len(text)}')
        #print(len(samples["text_output"]))

        image_embeds = self.ln_vision(self.visual_encoder(image))
        #print(f'image_embeds size after vision encoder: {image_embeds.size()}')
        #print(image_embeds)
        image_embeds = self.MCAN.img_feat_linear(image_embeds)  # Project image features to the correct size
        #print(f'image_embeds size after vision projection: {image_embeds.size()}')
        #print(image_embeds)
        #image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        #Tokenize text and pass through embedding layer
        #print(f'tokenize : { len(self.tokenizer) }')
        text_tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_txt_len).input_ids.to(image.device)
        #print(f'text_tokens size after tokenization : {text_tokens.size()}')
        #print(text_tokens)
        
        text_embeds = self.MCAN.embedding(text_tokens)
        #print(f'text_embeds size after tokenization and embedding: {text_embeds.size()}')
        #print(text_embeds)

        # Project text features to the correct size
        text_embeds,  _ = self.MCAN.lstm(text_embeds)#.view(-1, text_embeds.size(-1))).view(text_embeds.size(0), text_embeds.size(1), -1)
        #print(f'text_embeds size after projection: {text_embeds.size()}')
        #print(text_embeds)


        lang_feat_mask = self.MCAN.make_mask(text_tokens.unsqueeze(2))
        img_feat_mask = self.MCAN.make_mask(image_embeds)
        #print(f'lang_feat_mask size: {lang_feat_mask.size()}')
        #print(f'img_feat_mask size: {img_feat_mask.size()}')

        ##dat
        image_embeds = self.dat(image_embeds)

        # Using MCAN

        ##Encoder-Decoder
        lang_feat, img_feat = self.MCAN.backbone(text_embeds, image_embeds, lang_feat_mask, img_feat_mask)
        ##Flatten
        img_feat = self.MCAN.attflat_img(img_feat, img_feat_mask)
        lang_feat = self.MCAN.attflat_lang(lang_feat, lang_feat_mask)
        ##Normalization
        image_feats = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(lang_feat, dim=-1)

        ###============== Image-text Contrastive ===================###
        #print('========ITC=========')
        image_feats_all = concat_all_gather(image_feats)
        text_feat_all = concat_all_gather(text_feat)
        #print(f'image_feats_all size : {image_feats_all.size()}')
        #print(f'text_feat_all size : {text_feat_all.size()}')
        #print(f'image_feats : {image_feats.size()}')
        #print(f'text_feat size : {text_feat.size()}')

        ## CLIP의 ITC

        sim_i2t = torch.matmul(image_feats, text_feat_all.t())
        #sim_i2t = torch.matmul(image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)).squeeze()
        #sim_i2t, _ = sim_i2t.max(-1)
        #sim_i2t = sim_i2t / self.temp

        #print(f'sim_i2t size :{sim_i2t.size()}')

        sim_t2i = torch.matmul(text_feat, image_feats_all.t())
        #sim_t2i = torch.matmul(text_feat.unsqueeze(1), image_feats_all.unsqueeze(-1)).squeeze()#(text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)).squeeze()
        #sim_t2i, _ = sim_t2q.max(-1)
        #sim_t2i = sim_t2i / self.temp

        #print(f'sim_t2i size :{sim_t2i.size()}')

        ##이건 분산학습
        rank = 0 #dist.get_rank()

        bs = image.size(0) #batchsize
        
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(image.device)
        #print(f'target size : {targets.size()}')

        if "image_id" in samples.keys():
            image_ids = samples["image_id"].view(-1, 1)
            image_ids_all = concat_all_gather(image_ids)
            pos_idx = torch.eq(image_ids, image_ids_all.t()).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
            loss_itc = (loss_t2i + loss_i2t) / 2
        else:
            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2
            print(f'ITC LOSS : {loss_itc}')

        ###============== Image-text Matching ===================###
        #print("============ITM===============")

        # Gather all text input ids and attention masks across all GPUs
        text_input_ids_world = concat_all_gather(text_embeds)
        text_attention_mask_world = concat_all_gather(lang_feat_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)

        with torch.no_grad():
            if "image_id" in samples.keys():
                mask = torch.eq(image_ids, image_ids_all.t())
                sim_t2i.masked_fill_(mask, -10000)
                sim_i2t.masked_fill_(mask, -10000)
            else:
                sim_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # Select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        #print(f'image_embeds_neg size: {image_embeds_neg.size()}')

        # Select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        #print(f'text_ids_neg size: {text_ids_neg.size()}')
        #print(f'text_atts_neg size: {text_atts_neg.size()}')

        # Combine positive and negative samples
        text_ids_all = torch.cat([text_embeds, text_embeds, text_ids_neg], dim=0)
        text_atts_all = torch.cat([lang_feat_mask, lang_feat_mask, text_atts_neg], dim=0)
        #print(f'text_ids_all size: {text_ids_all.size()}')
        #print(f'text_atts_all size: {text_atts_all.size()}')

        # No query tokens needed, directly use text and image embeddings
        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], # pos, neg, pos
            dim=0)
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(image.device)

        #print(f'image_embeds_all size: {image_embeds_all.size()}')
        #print(f'image_atts_all size: {image_atts_all.size()}')

        # Generate masks using make_mask function from MCAN
        text_atts_all_mask = self.MCAN.make_mask(text_ids_all)
        image_atts_all_mask = self.MCAN.make_mask(image_embeds_all)

        # Use MCA_ED for ITM
        output_text, output_image = self.MCAN.backbone(
            text_ids_all,
            image_embeds_all,
            text_atts_all_mask,
            image_atts_all_mask
        )

        # output_text = self.MCAN.attflat_lang(
        #     output_text,
        #     text_atts_all_mask
        # )

        output_image = self.MCAN.attflat_img(
            output_image,
            image_atts_all_mask
        )

        # # proj_feat = output_text + output_image
        # # proj_feat = self.MCAN.proj_norm(proj_feat)
        # vl_embeddings = torch.sigmoid(self.MCAN.proj(proj_feat))
        vl_embeddings = output_image

        # Process the output from MCA_ED
        #vl_embeddings_text = output_text[:, :self.max_txt_len, :]
        #vl_embeddings_image = output_image[:, :image_embeds_all.size(1), :]

        # Assuming further processing is needed with only one of the outputs
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output
        #logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)
        print(f'ITM LOSS : {loss_itm}')


        ##================= Image Captioning ========================##
        ###============== Image Captioning ========================###
        #print("============IC===============")

        # Prepare decoder input ids and labels
        decoder_input_ids = text_tokens.clone().to(image.device)
        #print(f'decoder_input_ids size: {decoder_input_ids.size()}, dtype: {decoder_input_ids.dtype}')
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )


        # Embed decoder input ids
        decoder_input_embeds = self.MCAN.embedding(decoder_input_ids)
        decoder_input_embeds, _ = self.MCAN.lstm(decoder_input_embeds)

        #print(f'decoder_input_embeds size: {decoder_input_embeds.size()}, dtype: {decoder_input_embeds.dtype}')

        
        # Use MCA_ED for IC
        output_text, _ = self.MCAN.backbone(
            decoder_input_embeds,
            image_embeds,
            lang_feat_mask,
            img_feat_mask
        )

        #print(f'output_text size: {output_text.size()}, dtype: {output_text.dtype}')


        # Pass the output text embeddings to the language modeling head
        logits = self.lm_head(output_text)
        #print(f'logits size: {logits.size()}, dtype: {logits.dtype}')
        #print(f'logits: {logits}')

        # Verify logits and labels range
        #print(f'logits min: {logits.min()}, max: {logits.max()}')
        #print(f'labels min: {labels.min()}, max: {labels.max()}')

        # Ensure logits are in float32 for cross_entropy
        logits = logits.float()
        
        # Calculate the language modeling loss
        loss_lm = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        print(f'IC LOSS: {loss_lm.item()}')

        # Combine losses
        total_loss = loss_itc + loss_itm + loss_lm
        print(f'Total loss: {total_loss.item()}')

        return BlipOutput(
            loss=total_loss,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = self.MCAN.img_feat_linear(image_embeds)  # Project image features to the correct size

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )

        outputs = self.MCAN.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = self.MCAN.img_feat_linear(image_embeds)  # Project image features to the correct size
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        lang_feat_mask = self.MCAN.make_mask(image_embeds)

        query_output = self.MCAN.backbone(
            image_embeds,
            image_embeds,
            lang_feat_mask,
            lang_feat_mask
        )
        return query_output, image_embeds

    def forward_text(self, text_tokens):
        text_embeds = self.MCAN.embedding(text_tokens.input_ids.to(self.device))
        lang_feat_mask = self.MCAN.make_mask(text_tokens.input_ids.unsqueeze(2))
        text_embeds, _ = self.MCAN.lstm(text_embeds)

        text_output = self.MCAN.backbone(
            text_embeds,
            text_embeds,
            lang_feat_mask,
            lang_feat_mask
        )
        return text_output[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_inputs = self.MCAN.img_feat_linear(image_inputs)  # Project image features to the correct size
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(image_inputs.device)
        text_embeds = self.MCAN.embedding(text_ids.input_ids.to(image_inputs.device))
        text_embeds, _ = self.MCAN.lstm(text_embeds)
        lang_feat_mask = self.MCAN.make_mask(text_ids.input_ids.unsqueeze(2))

        output_itm = self.MCAN.backbone(
            text_embeds,
            image_inputs,
            lang_feat_mask,
            image_atts
        )
        vl_embeddings = output_itm[:, :text_embeds.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        image = samples.get("image")
        caption = samples.get("text_input")

        assert mode in ["image", "text", "multimodal"], "mode must be one of 'image', 'text', 'multimodal'"

        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert image is not None, "Image is not provided for mode 'image' or 'multimodal'"
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_embeds_frozen = self.MCAN.img_feat_linear(image_embeds_frozen)  # Project image features to the correct size
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            lang_feat_mask = self.MCAN.make_mask(image_embeds_frozen)

            query_output = self.MCAN.backbone(
                image_embeds_frozen,
                image_embeds_frozen,
                lang_feat_mask,
                lang_feat_mask
            )
            image_embeds = query_output
            image_features = F.normalize(image_embeds, dim=-1)

        elif mode == "text":
            assert caption is not None, "text input is None for mode 'text' or 'multimodal'"

            text_tokens = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
            text_embeds = self.MCAN.embedding(text_tokens.input_ids)
            text_embeds, _ = self.MCAN.lstm(text_embeds)
            lang_feat_mask = self.MCAN.make_mask(text_tokens.input_ids.unsqueeze(2))

            text_output = self.MCAN.backbone(
                text_embeds,
                text_embeds,
                lang_feat_mask,
                lang_feat_mask
            )
            text_embeds = text_output
            text_features = F.normalize(text_embeds, dim=-1)

        elif mode == "multimodal":
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_embeds_frozen = self.MCAN.img_feat_linear(image_embeds_frozen)  # Project image features to the correct size
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            lang_feat_mask = self.MCAN.make_mask(image_embeds_frozen)

            text_tokens = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
            text_embeds = self.MCAN.embedding(text_tokens.input_ids)
            text_embeds, _ = self.MCAN.lstm(text_embeds)
            text_feat_mask = self.MCAN.make_mask(text_tokens.input_ids.unsqueeze(2))

            output = self.MCAN.backbone(
                text_embeds,
                image_embeds_frozen,
                text_feat_mask,
                lang_feat_mask
            )

            multimodal_embeds = output[:, :text_embeds.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)