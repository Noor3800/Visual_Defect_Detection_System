import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_model(num_classes: int = 2):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze the pretrained feature extractor initially.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
