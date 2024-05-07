import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as transforms_v2
import numpy as np
from torch.hub import load_state_dict_from_url
from typing import Any

urls = [
    'https://github.com/andreasHovaldt/custom-pytorch-models/raw/rizznet50/models_weights/dual_resnet50pose_0.000313_rgb.pth',
    'https://github.com/andreasHovaldt/custom-pytorch-models/raw/rizznet50/models_weights/dual_resnet50pose_0.000313_depth.pth',
    'https://github.com/andreasHovaldt/custom-pytorch-models/raw/rizznet50/models_weights/dual_resnet50pose_0.000313_final.pth.pth',
]

# Custom transform to threshold the depth image
class Threshholder(nn.Module):
    def forward(self, img: np.ndarray):
        assert type(img) == np.ndarray
        img[img > 1] = 0
        img = torch.from_numpy(img).unsqueeze(0)
        return img


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
    
    
    # RGB and Depth transforms
    # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
    @property
    def rgb_transform(self) -> transforms_v2.Compose:
        return transforms_v2.Compose([
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize((224,224), interpolation=transforms_v2.InterpolationMode.BILINEAR, antialias=True),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @property
    def depth_transform(self) -> transforms_v2.Compose:
        return transforms_v2.Compose([
            Threshholder(),
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize((224,224), interpolation=transforms_v2.InterpolationMode.BILINEAR, antialias=True),
            transforms_v2.Grayscale(num_output_channels=3), # Converts image from 1 to 3 channels
        ])


def custom_dual_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomDualResnet50:
    
    model = CustomDualResnet50(**kwargs)
    
    if pretrained:
        state_dict_rgb = load_state_dict_from_url(url=urls[0], progress=progress)
        model.rgb_model.load_state_dict(state_dict_rgb)
        
        state_dict_depth = load_state_dict_from_url(url=urls[1], progress=progress)
        model.depth_model.load_state_dict(state_dict_depth)
        
        state_dict_final = load_state_dict_from_url(url=urls[2], progress=progress)
        model.final_pose_estimation.load_state_dict(state_dict_final)
    
    return model