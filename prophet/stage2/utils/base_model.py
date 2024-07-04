from omegaconf import OmegaConf
import torch
import os

class BaseModel:
    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xxl": "/root/workspace/EunJuPark/24s-VQA-MLLM/outputs/model/daiv_flant5xxl.yaml",
    }

    @classmethod
    def from_pretrained(cls, model_type, *model_args, **kwargs):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), f"Unknown model type {model_type}"
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls(model_cfg, *model_args, **kwargs)
        model.load_state_dict(torch.load(cls.default_weight_path(model_type)))
        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), f"Unknown model type {model_type}"
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    @classmethod
    def default_weight_path(cls, model_type):
        return cls.default_config_path(model_type).replace(".yaml", ".pth")

def get_abs_path(rel_path):
    from .registry import registry
    return os.path.join(registry.get_path("library_root"), rel_path)
