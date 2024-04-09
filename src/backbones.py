import torch.nn as _nn
import torchvision.models as _models


def resnet18_backbone(pretrained=False):
    weights = _models.ResNet18_Weights.DEFAULT if pretrained else None
    resnet = _models.resnet18(weights=weights, num_classes=1000)
    num_features = resnet.fc.in_features
    resnet.fc = _nn.Identity()
    return resnet, num_features


def resnet50_backbone(pretrained=False):
    weights = _models.ResNet50_Weights.DEFAULT if pretrained else None
    resnet = _models.resnet50(weights=weights, num_classes=1000)
    num_features = resnet.fc.in_features
    resnet.fc = _nn.Identity()
    return resnet, num_features


def dummy_backbone():
    net = _nn.Sequential(_nn.Conv2d(3, 1, 4, 4), _nn.Flatten())
    num_features = 3136
    return net, num_features


def get_backbone(backbone_name):
    if backbone_name == "resnet18":
        backbone = resnet18_backbone(pretrained=True)
    elif backbone_name == "resnet50":
        backbone = resnet50_backbone(pretrained=True)
    elif backbone_name == "dummy":
        backbone = dummy_backbone()
    else:
        raise ValueError(f"Unknown backbone {backbone_name}")
    return backbone
