import torch.nn as nn
from torchvision import models

def _fc(in_features:int, num_classes:int, hidden:int|None=None):
    if not hidden:
        return nn.Linear(in_features, num_classes)
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(hidden, num_classes),
    )

def build_model(num_classes:int, backbone:str="resnet18", pretrained:bool=True, head_hidden:int|None=None):
    bb = backbone.lower()

    if bb == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_f = net.fc.in_features
        net.fc = _fc(in_f, num_classes, head_hidden)
        return net

    if bb == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = models.convnext_tiny(weights=weights)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = _fc(in_f, num_classes, head_hidden)
        return net

    if bb == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = _fc(in_f, num_classes, head_hidden)
        return net

    # default
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    in_f = net.fc.in_features
    net.fc = _fc(in_f, num_classes, head_hidden)
    return net
