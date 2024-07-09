import torch
import torch.nn as nn
from transformers import BeitFeatureExtractor, BeitForImageQuestionAnswering, AutoTokenizer
import logging

logging.set_verbosity_error()

class BEiT3_VQA(nn.Module):
    def __init__(self, model_name_or_path, answer_size):
        super().__init__()

        # Load BEiT3ForVisualQuestionAnswering model and feature extractor
        self.model = BeitForImageQuestionAnswering.from_pretrained(model_name_or_path)
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.answer_size = answer_size

        # Additional layers for VQA task
        self.proj_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.proj = nn.Linear(self.model.config.hidden_size, answer_size)

    def forward(self, img, ques_text):
        # Extract image features using BEiT's feature extractor
        image = self.feature_extractor(images=img, return_tensors="pt")

        # Tokenize and encode question text
        inputs = self.tokenizer(ques_text, return_tensors="pt", padding=True, truncation=True)

        # Forward pass through BEiT model
        outputs = self.model(pixel_values=image.pixel_values, **inputs)

        # Extract pooled output
        pooled_output = outputs.logits[:, :self.answer_size]

        # Apply layer normalization
        pooled_output = self.proj_norm(pooled_output)

        # Project pooled output to answer space
        proj_output = self.proj(pooled_output)

        return proj_output
