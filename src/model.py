from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    ResNet50_Weights,
    convnext_tiny,
    densenet121,
    resnet50,
)


MODEL_NAMES = ("resnet50", "densenet121", "convnext_tiny")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def create_model(model_name: str, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(model.fc.in_features, 1))
    elif model_name == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(model.classifier.in_features, 1))
    elif model_name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(0.35), nn.Linear(in_features, 1))
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {MODEL_NAMES}.")
    return model


def freeze_backbone(model: nn.Module, model_name: str):
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.fc if model_name == "resnet50" else model.classifier
    for parameter in head.parameters():
        parameter.requires_grad = True


def unfreeze_top_stage(model: nn.Module, model_name: str):
    if model_name == "resnet50":
        stage = model.layer4
    elif model_name == "densenet121":
        stage = model.features.denseblock4
        for parameter in model.features.norm5.parameters():
            parameter.requires_grad = True
    elif model_name == "convnext_tiny":
        stage = model.features[-1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    for parameter in stage.parameters():
        parameter.requires_grad = True


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = create_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
