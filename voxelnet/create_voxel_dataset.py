import voxel_maker
import os 
import numpy as np 
from tqdm import tqdm

input_dir = "/home/omniverse-11/Desktop/Datasets/voxel_dataset/depth/"
destination_dir = "/home/omniverse-11/Desktop/Datasets/voxel_dataset/voxel/"
# os.makedirs(destination_dir)

for depth_img in tqdm(os.listdir(input_dir)):
    img = np.load(os.path.join(input_dir, depth_img))
    voxel = voxel_maker.create_np_voxel_from_depth_image(img)

    voxel_name = f"voxel_{depth_img.split('_')[2]}"

    np.save(os.path.join(destination_dir, voxel_name), voxel)
