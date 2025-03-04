import torch
import numpy as np
import torchinfo

from voxelnet import VoxelNet

voxelnet = VoxelNet()

voxelnet.load_state_dict(torch.load("voxelnet/voxelnet_l_5.596930714091286e-05_ep_49.pth", map_location=torch.device('cpu')))   

voxel = np.load("/home/dreezy/Downloads/voxel_122.npy")
voxel_tensor = torch.tensor(voxel).float()
#print(voxel_tensor.shape)
torchinfo.summary(voxelnet, 
                  input_data=(torch.randn(1, 1, 270, 110, 40)),
                  col_names=["input_size", "output_size", "num_params", "trainable"],
                  col_width=20,
                  row_settings=["var_names"]
)