import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from typing import Any

class CustomEfficientNet_V2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the pretrained EfficientNetV2-S model
        self.model = models.efficientnet_v2_s()

        # Define new classifying layer (From classification task to regression task)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=6, bias=True),
            nn.Flatten(),
        )
        
        # Freeze the feature parameters
        for param in self.model.features.parameters():
            param.requires_grad = False
        

    def forward(self, x):
        return self.model(x)


def custom_efficientnet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomEfficientNet_V2:
    
    model = CustomEfficientNet_V2(**kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(url='https://github.com/ES-24-ROB6-662/SeasonyCNN/raw/Dataset-loading/models_weights/efficientnetv2_loss0.000124.pth', progress=progress)
        model.load_state_dict(state_dict)
    
    return model