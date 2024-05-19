from cv2 import transform
import torch
import numpy 
from torch import nn
from torch.nn.modules import Sequential
from torch.utils.data import Dataset
import pandas as pd 
import os 

import numpy as np

# defining the model 

class VoxelNet(nn.Module):
    def __init__(self):
        super(VoxelNet, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 256, kernel_size=21, bias=False,stride=7),
            #nn.BatchNorm3d(256),
            nn.LeakyReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, bias=False),
            #nn.BatchNorm3d(512),
            nn.LeakyReLU(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv3d(512, 512, kernel_size=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv3d(512, 256, kernel_size=1, bias=True),
            nn.LeakyReLU(),

        )
        self.fc = nn.Sequential(
            nn.LazyLinear(6)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        residual = x
        x = self.conv_layer3(x)
        x = x + residual
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


class VoxelDataset(Dataset):
    def __init__(self, annotations_file, voxel_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.voxel_dir = voxel_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        voxel_name = (self.img_labels.iloc[idx, 0]).replace('rgb_image_', 'voxel_').replace('.png','.npy')
        voxel_path = os.path.join(self.voxel_dir, voxel_name)
        voxel = np.load(voxel_path)
        
        # add random noise to avoide hyper focused weights. this is done here to make it unique for each epoch   
        noise = np.random.uniform(low = 0, high= 1, size=(voxel.shape))
        voxel[noise > 0.95] = 1
        
        voxel = np.expand_dims(voxel, axis = 0) # add channel 
        voxel = torch.tensor(voxel).float() # convert to tensor 
        
        label = torch.tensor(self.img_labels.iloc[idx, 1:7])

        return voxel, label


class PoseLoss():
    def __init__(self, unit_balancer):
        self.unit_balancer = unit_balancer
        self.mse_loss = torch.nn.MSELoss()
    def loss(self, output, target):
        transformed_output = output * self.unit_balancer
        transformed_target = target * self.unit_balancer
        loss = self.mse_loss(transformed_output, transformed_target)
        return loss


