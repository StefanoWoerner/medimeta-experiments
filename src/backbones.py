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