import torch.nn as nn
from torchvision import models

def get_efficientnet(num_classes):
    model = models.efficientnet_b0(pretrained=True)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    return model
