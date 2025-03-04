import open3d as o3d 
import numpy as np 
import torch 
import cv2 

import matplotlib.pyplot as plt 


MIN_BOUNDS = np.array([0,0,0])
MAX_BOUNDS = np.array([195,104,55])


def make_PCD(depth_image, threshold=0.7, intrinsics=[1280, 800, 241.69, 241.69,651.99,412.456421]):
    '''
    A function that converts an RGB_image + depth image to a pointcloud.

    Parameters:
    depth_image: A depth image as a (height, width, 1) tensor
    threshold: A threshold integer in meters. The points with depths above this value will NOT be shown.
    intrinsics: An array containing the fx, fy, cx, and cy intrinsic parameters for the camera.
    '''
    height = intrinsics[0]
    width = intrinsics[1]
    fx = intrinsics[2]
    fy = intrinsics[3]
    cx = intrinsics[4]
    cy = intrinsics[5]

    # Normalize depth image

    # plt.imshow(depth_image)
    # plt.show()

    depth_image[depth_image == np.inf] = 0
    depth_image[depth_image > threshold] = 0
    # print(depth_image.shape)
    # Convert depth image back to open3d format
    depth_image_open3d = o3d.geometry.Image(depth_image.astype(np.float32))

    # Create RGBD image from color and depth
    # Set intrinsic camera parameters
    intrinsics = o3d.camera.PinholeCameraIntrinsic(height, width, fx, fy, cx, cy)

    # return the point cloud from RGBD image and intrinsic parameters
    return o3d.geometry.PointCloud.create_from_depth_image(depth_image_open3d, intrinsics)


def voxel_from_image(depth_image, voxel_size):
    pc = make_PCD(depth_image = depth_image)
    # pc, ind = pc.remove_statistical_outlier(nb_neighbors=20,
    #                               std_ratio=3.0)
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size)
    return voxel


def create_np_voxel_from_depth_image(depth_image, voxel_size = 0.01):
    o3d_voxel = voxel_from_image(depth_image, voxel_size)
    voxel_array = []
    for point in o3d_voxel.get_voxels():
        voxel_array.append(point.grid_index)
    voxel_array = np.array(voxel_array)
    voxel_matrix = np.zeros(tuple(MAX_BOUNDS))

    voxel_array[voxel_array[:,0] > MAX_BOUNDS[0]-1, :] = 0 
    voxel_array[voxel_array[:,1] > MAX_BOUNDS[1]-1, :] = 0 
    voxel_array[voxel_array[:,2] > MAX_BOUNDS[2]-1, :] = 0 

    voxel_matrix[voxel_array[:,0], voxel_array[:,1], voxel_array[:,2]] = 1 
    voxel_matrix[0,0,0] = 0
    return voxel_matrix.astype(np.uint8)



if __name__ == "__main__":    
    depth_image = np.load("/home/a/seasony/testing-dataset-rots/depth/depth_image_1.npy")
    rgb_image = cv2.imread("/home/a/seasony/testing-datasetWithProcessedDepth/rgb/rgb_image_0.png")
    # print(rgb_image.shape)
    #
    # voxel = voxel_from_image(depth_image, 0.01)
    # voxel_list = []
    # for vox in voxel.get_voxels():
    #     # print(vox.grid_index)
    #     voxel_list.append(vox.grid_index)
    #
    # voxel_array = np.array(voxel_list)
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    # # Scatter plot of voxel coordinates
    # ax.scatter(voxel_array[:, 0], voxel_array[:, 1], voxel_array[:, 2])
    # 
    # print(f"i0 max {np.max(voxel_array[:,0])} min {np.min(voxel_array[:,0])} \ni1 max {np.max(voxel_array[:,1])} min {np.min(voxel_array[:,1])} \ni1 max {np.max(voxel_array[:,2])} min {np.min(voxel_array[:,2])}")
    #
    # voxel_matrix = np.zeros(tuple(MAX_BOUNDS))
    #
    # voxel_array[voxel_array[:,0] > MAX_BOUNDS[0]-1, :] = 0 
    # voxel_array[voxel_array[:,1] > MAX_BOUNDS[1]-1, :] = 0 
    # voxel_array[voxel_array[:,2] > MAX_BOUNDS[2]-1, :] = 0 
    #
    # print(voxel_array)
    # voxel_matrix[voxel_array[:,0], voxel_array[:,1], voxel_array[:,2]] = 1 
    #
    # print(voxel_matrix)
    
    voxel_matrix = create_np_voxel_from_depth_image(depth_image, 0.01)

    test_voxel_array = []
    for m in range(270):
        for n in range(110):
            for k in range(40):
                if voxel_matrix[m,n,k] == 1: 
                    test_voxel_array.append(np.array([m, n, k]))
    test_voxel_array = np.array(test_voxel_array)   
    ax.scatter(test_voxel_array[:, 0], test_voxel_array[:, 1], test_voxel_array[:, 2])
    # ax.voxels(voxel_matrix, edgecolor='k') 


    plt.show()
