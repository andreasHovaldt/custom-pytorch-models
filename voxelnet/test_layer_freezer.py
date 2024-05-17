import torch
import numpy 
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd 
import os 

import numpy as np

from voxelnet import VoxelNet 
from voxelnet import VoxelDataset
from voxelnet import PoseLoss 

voxelnet = VoxelNet()

layer_to_freeze = np.random.randint(0,5)
print(layer_to_freeze)
for layer, param in enumerate(voxelnet.parameters()):
    if layer == layer_to_freeze:
        param.requires_grad = False 
    else:
        param.requires_grad = True
    print(layer, param.requires_grad)

