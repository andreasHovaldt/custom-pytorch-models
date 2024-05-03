import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from typing import Any

class CustomDualResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the pretrained EfficientNetV2-S model
        self.rgb_model = models.resnet50()
        self.depth_model = models.resnet50()

        # Define new classifying layer (From classification task to regression task)
        self.rgb_model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=2048, out_features=100, bias=True),
            nn.Tanh(),
        )
        self.depth_model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=2048, out_features=100, bias=True),
            nn.Tanh(),
        )
        self.final_pose_estimation = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=200, out_features=6, bias=True),
        )
        
        

    def forward(self, rgb, depth):
        rgb = self.rgb_model(rgb)
        depth = self.depth_model(depth)
        return self.final_pose_estimation(torch.cat([rgb, depth], dim=1))


def custom_dual_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomDualResnet50:
    
    model = CustomDualResnet50(**kwargs)
    
    if pretrained:
        state_dict_rgb = load_state_dict_from_url(url='https://github.com/andreasHovaldt/custom-pytorch-models/raw/rizznet50/models_weights/dual_resnet50pose_rgb.pth', progress=progress)
        model.rgb_model.load_state_dict(state_dict_rgb)
        
        state_dict_depth = load_state_dict_from_url(url='https://github.com/andreasHovaldt/custom-pytorch-models/raw/rizznet50/models_weights/dual_resnet50pose_depth.pth', progress=progress)
        model.depth_model.load_state_dict(state_dict_depth)
        
        state_dict_final = load_state_dict_from_url(url='https://github.com/andreasHovaldt/custom-pytorch-models/raw/rizznet50/models_weights/dual_final_pose_estimator.pth', progress=progress)
        model.final_pose_estimation.load_state_dict(state_dict_final)
    
    return model