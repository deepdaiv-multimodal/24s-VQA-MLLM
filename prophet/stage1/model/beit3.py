from torch import nn
from torchscale.model import BEiT3
from modeling_finetune import BEiT3ForVisualQuestionAnswering

class BEiT3Model(nn.Module):
    def __init__(self, __C, answer_size):
        super(BEiT3Model, self).__init__()
        self.beit3 = BEiT3ForVisualQuestionAnswering(
            args=__C,
            num_classes=answer_size,
            vocab_size=__C.VOCAB_SIZE,
            img_size=__C.IMG_SIZE,
            patch_size=__C.PATCH_SIZE,
            drop_path_rate=__C.DROP_PATH_RATE,
            checkpoint_activations=__C.CHECKPOINT_ACTIVATIONS,
            encoder_embed_dim=__C.ENCODER_EMBED_DIM,
            multiway=__C.multiway
        )

    def forward(self, x):
        return self.beit3(x)
