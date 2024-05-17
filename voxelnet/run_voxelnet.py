from voxelnet import VoxelNet
from voxelnet import VoxelDataset 
from torch.utils.data import DataLoader
import numpy as np 
import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm

#first we load in a voxel 



voxel = np.load("/home/a/seasony/dataset_depth-10k/dataset/voxels/voxel_0.npy")

voxel = np.expand_dims(voxel,0)

voxel = np.expand_dims(voxel, 0)

print(voxel.shape)

voxel_tensor = torch.tensor(voxel).float()

labels_path = "/home/a/seasony/testing-dataset-rots/labels.csv"

voxel_dir = "/home/a/seasony/testing-dataset-rots/voxel/"

voxel_dataset = VoxelDataset(labels_path, voxel_dir)

voxel_dataloader = DataLoader(voxel_dataset)

voxelnet = VoxelNet()

voxelnet.load_state_dict(torch.load("/home/a/seasony/custom-pytorch-models/voxelnet/voxelnet_l_0.00019675052317325026_ep_99.pth", map_location=torch.device('cpu')))

results = []
for voxel, label in tqdm(voxel_dataloader):
    with torch.no_grad():
        out = voxelnet(voxel)
        buffer = np.zeros(12)
        buffer[0:6] = np.array(label)
        buffer[6:12] = np.array(out)
        results.append(buffer)

results = np.array(results)


colors = ["red", "green", "blue"]

fig, axs = plt.subplots(1,3)
for n in range(3):
    axs[n].scatter(results[:,n+3], results[:,n+6+3], c = colors[n])
plt.show()
