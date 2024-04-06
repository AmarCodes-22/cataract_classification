from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import CLIPVisionModel


@dataclass
class ClassificationModelOutput:
    logits: torch.Tensor


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)


class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32", num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path)
        self.backbone_output_size = self.backbone.config.hidden_size

        self.head = LinearHead(input_dim=self.backbone_output_size, output_dim=self.num_classes)

    def forward(self, x) -> ClassificationModelOutput:
        x = self.backbone(x).pooler_output
        x = self.head(x)
        return ClassificationModelOutput(logits=x)


class ExportWrapper(nn.Module):
    def __init__(self, classification_model: ClassificationModel):
        super().__init__()

        self.model = classification_model

    @torch.inference_mode()
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        with autocast(enabled=True):
            x = x / 255
            x = self.model(x).logits
            x = torch.softmax(x, dim=1)
        return x
