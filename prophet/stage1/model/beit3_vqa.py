import torch
from torch import nn
from transformers import BeitModel, BeitConfig

class BEiT3ForVisualQuestionAnswering(nn.Module):
    def __init__(self, model_name, num_classes, norm_layer=nn.LayerNorm):
        super(BEiT3ForVisualQuestionAnswering, self).__init__()
        self.config = BeitConfig.from_pretrained(model_name)
        self.beit3 = BeitModel.from_pretrained(model_name)
        embed_dim = self.config.hidden_size
        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            norm_layer(embed_dim),
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), 
            norm_layer(embed_dim * 2), 
            nn.GELU(), 
            nn.Linear(embed_dim * 2, num_classes), 
        )
        self.head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, image, question, padding_mask, **kwargs):
        outputs = self.beit3(
            input_ids=question, 
            attention_mask=padding_mask,
            pixel_values=image
        )
        x = outputs.last_hidden_state
        cls_rep = self.pooler(x[:, 0])
        return self.head(cls_rep)
