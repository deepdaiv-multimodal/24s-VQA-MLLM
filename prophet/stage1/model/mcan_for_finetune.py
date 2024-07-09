import torch
import torch.nn as nn
from transformers import AutoModel
from .mcan import MCA_ED, AttFlat
import os

class MCANForFinetune(nn.Module):
    def __init__(self, __C, answer_size, base_answer_size=3129):
        super().__init__()
        self.__C = __C

        # BERT 모델 초기화
        self.bert = AutoModel.from_pretrained(__C.BERT_VERSION)
        self.img_feat_linear = nn.Linear(768, __C.HIDDEN_SIZE, bias=False)  # 768로 명확히 지정
        self.lang_adapt = nn.Linear(__C.LANG_FEAT_SIZE, __C.HIDDEN_SIZE)
        
        # MCA_ED Backbone 초기화
        self.backbone = MCA_ED(__C)
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.proj_norm = nn.LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, base_answer_size)
        self.proj1 = nn.Linear(__C.FLAT_OUT_SIZE, answer_size - base_answer_size)

    @torch.no_grad()
    def parameter_init(self):
        self.proj1.weight.data.zero_()
        self.proj1.bias.data = self.proj.bias.data.mean() + torch.zeros(self.proj1.bias.data.shape)

    def forward(self, input_tuple, output_answer_latent=False):
        img_feat, ques_ix = input_tuple

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = None

        # Pre-process Language Feature
        lang_feat = self.bert(
            ques_ix, 
            attention_mask=~lang_feat_mask.squeeze(1).squeeze(1)
        )[0]
        lang_feat = self.lang_adapt(lang_feat)

        # Pre-process Image Feature
        batch_size, num_features, feature_dim = img_feat.size()  # Get dimensions
        img_feat = img_feat.view(batch_size * num_features, feature_dim)  # Flatten img_feat to 2D
        img_feat = self.img_feat_linear(img_feat)  # Apply linear layer
        img_feat = img_feat.view(batch_size, num_features, -1)  # Reshape back to 3D

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )
        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        answer_latent = self.proj_norm(proj_feat)
        proj_feat = torch.cat((self.proj(answer_latent), self.proj1(answer_latent)), dim=-1)

        if output_answer_latent:
            return proj_feat, answer_latent

        return proj_feat

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

def load_checkpoint_with_different_shape(ckpt_path, model):
    state_dict = torch.load(ckpt_path)
    model_state_dict = model.state_dict()

    for name, param in state_dict.items():
        if name in model_state_dict:
            if param.shape != model_state_dict[name].shape:
                print(f"Transforming parameter: {name}, from shape: {param.shape} to shape: {model_state_dict[name].shape}")

                # Transform the weights using a linear layer
                if 'img_feat_linear' in name:
                    in_features = param.shape[1]
                    out_features = model_state_dict[name].shape[0]
                    linear = nn.Linear(in_features, out_features, bias=False)
                    with torch.no_grad():
                        linear.weight.copy_(param)
                    param = linear.weight

                state_dict[name] = param

    model.load_state_dict(state_dict, strict=False)