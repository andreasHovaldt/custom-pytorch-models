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

from tqdm import tqdm

label_path = "/home/omniverse-11/Desktop/Datasets/voxel_dataset/labels.csv"
voxel_dir = "/home/omniverse-11/Desktop/Datasets/voxel_dataset/voxel/"

dataset = VoxelDataset(label_path, voxel_dir)

voxel_dataloader = DataLoader(dataset, batch_size=25, num_workers = 8, prefetch_factor = 400)

voxelnet = VoxelNet()
voxelnet.to('cuda')
# voxelnet.load_state_dict(torch.load("voxelnet_rot_l_0.015363795682787895_ep_99.pth"))

loss_function = PoseLoss( torch.tensor([1,1,2,0.02,0.02,0.02]).cuda() )

optimizer = torch.optim.Adam(voxelnet.parameters(), lr = 0.00001, weight_decay= 0.0001)
losses = []
for ep in range(50):
    #freeze a random layer hehe 
    # layer_to_freeze = np.random.random_integers(0,5)
    # print(f"freezing layer {layer_to_freeze}")
    # for layer, param in enumerate(voxelnet.parameters()):
    #     if layer == layer_to_freeze:
    #         param.requires_grad = False 
    #     else:
    #         param.requires_grad = True

    for voxel, label in tqdm(voxel_dataloader):
        pred = voxelnet(voxel.cuda())
        loss = loss_function.loss(pred, label.float().cuda())
        #print(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    os.system('cls||clear')
    print(f"loss {loss.item()}, ep {ep}")

torch.save(voxelnet.state_dict(), f"voxelnet_rot_l_{loss.item()}_ep_{ep}.pth")
np.savetxt("losses.txt",np.array(losses))
