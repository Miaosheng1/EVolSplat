import numpy as np
import os
import imageio.v2 as imageio
import cv2
import open3d as o3d
from typing import Literal

from copy import deepcopy

X_MIN, X_MAX = -20, 20
Y_MIN, Y_MAX = -20, 4.8
Z_MIN, Z_MAX = -20, 70

class WaymoPCDGenerator():
    # "Drop25" discards the last frame out of every 4 frames
    sparsity: Literal['Drop90','Drop50',"Drop80","Drop25","full"] = "full"
    use_bbx: bool = True
    """Accumulate the number of pointcloud frames"""

    def __init__(self,spars='full',save_dir="Drop50",frame_start=0, filer_sky=True,depth_cosistency=False) -> None:
        self.sparsity = spars
        self.save_dir = save_dir
        self.cam2world_dict_00 = None
        self.dir_name = None
        self.frame_start = frame_start
        self.filter_sky = filer_sky
        self.depth_cosistency = depth_cosistency

    
    def get_bbx(self):
        return np.array([X_MIN, Y_MIN, Z_MIN]), np.array([X_MAX, Y_MAX, Z_MAX])

    
    def crop_pointcloud(self,bbx_min, bbx_max, points, color):
        mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2]+50)

        return points[mask], color[mask]
    
    def split_pointcloud(self,bbx_min, bbx_max, points, color):
        mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])

        inside_pnt, inside_rgb = points[mask], color[mask]
        outside_pnt, outside_rgb = points[~mask], color[~mask]
        return inside_pnt,inside_rgb,outside_pnt,outside_rgb

    def forward(self, dir_name, poses, intrinsics, H,W,down_scale=2):

        self.dir_name = dir_name
        self.depth_dir = os.path.join(self.dir_name, 'depth')
   

        depth_files = []
        selected_index = []
        self.c2w = []
        self.intri = []
        self.H, self.W = H, W
        
        """ Process differnt Sparsity level KITTI-360 """
        for i, file_name in enumerate(sorted(os.listdir(self.depth_dir))):
            if self.sparsity == "Drop50":
               if i % 4 == 2 or i % 4 == 3:
                     continue
            elif self.sparsity == 'Drop80':
                if i % 5 != 0:
                    continue 
            elif self.sparsity == 'Drop25':
                if i % 4 == 2:
                    continue 
            elif self.sparsity == 'Drop90':
                if i % 10 != 0:
                    continue
        
            depth_files.append(file_name)
            ## The pose for projection must be in OpenCV coordinate system
            self.c2w.append(poses[i]*np.array([1, -1, -1, 1]))
            self.intri.append(intrinsics[i])
            selected_index.append(i)

        print(f"{self.sparsity} :ALL frames: {len(depth_files)}:,{selected_index} \n")

      
        if self.depth_cosistency:
            cosistency_mask = self.depth_cosistency_check(depth_files=depth_files)
        else:
            cosistency_mask = [np.ones((H,W)) for _ in range(len(depth_files))]

        accmulat_pointcloud = self.accumulat_pcd(depth_files=depth_files,cosistency_mask=cosistency_mask,down_scale=down_scale)

        """ Output and Save the .ply pointcloud """
        os.makedirs(os.path.join(self.dir_name, self.save_dir),exist_ok=True)

        points = accmulat_pointcloud[:, :3]
        colors = accmulat_pointcloud[:, 3:]

        if self.use_bbx:
            bbx_min, bbx_max = self.get_bbx()
            print(f"BBX Range: {bbx_min},{bbx_max} \n")
            points, colors = self.crop_pointcloud(bbx_min, bbx_max, points, colors)
            inside_pnt,inside_rgb,outside_pnt,outside_rgb = self.split_pointcloud(bbx_min, bbx_max, points, colors)

            ## inside filter
            inside_pointcloud = o3d.geometry.PointCloud()
            inside_pointcloud.points = o3d.utility.Vector3dVector(inside_pnt[:, :3])
            inside_pointcloud.colors = o3d.utility.Vector3dVector(inside_rgb)
            cl, ind = inside_pointcloud.remove_statistical_outlier(nb_neighbors=35,std_ratio=1.5)
            inside_pointcloud = inside_pointcloud.select_by_index(ind)

            ## outside filter
            outside_pointcloud = o3d.geometry.PointCloud()
            outside_pointcloud.points = o3d.utility.Vector3dVector(outside_pnt[:, :3])
            outside_pointcloud.colors = o3d.utility.Vector3dVector(outside_rgb)
            cl, ind = outside_pointcloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
            outside_pointcloud = outside_pointcloud.select_by_index(ind)

            combined_pointcloud = inside_pointcloud + outside_pointcloud
            combined_pointcloud = combined_pointcloud.uniform_down_sample(every_k_points=2)
        
            o3d.io.write_point_cloud(os.path.join(self.dir_name, self.save_dir, f'{self.frame_start}.ply'), combined_pointcloud)
            print(f"Save the pointcloud in {os.path.join(self.dir_name, self.save_dir)} !")

        else:

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            ## Filter the noisy pointcloud and downsample the pcd in 3D space
            cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=30,std_ratio=1.5)
            point_cloud = point_cloud.select_by_index(ind)
            point_cloud = point_cloud.uniform_down_sample(every_k_points=3)
        
            o3d.io.write_point_cloud(os.path.join(self.dir_name, self.save_dir, f'{self.frame_start}.ply'), point_cloud)
            print(f"Save the pointcloud in {os.path.join(self.dir_name, self.save_dir)} !")




    """The accumulated pointcloud locate in the OpenCV system """
    def accumulat_pcd(self,depth_files,cosistency_mask,down_scale:int = 2):
        color_pointclouds = []

        for i, file_name in enumerate(depth_files):
            depth_file = os.path.join(self.depth_dir, file_name)
            depth = np.load(depth_file)  # (376, 1408)
            rgb_file = os.path.join(self.dir_name, file_name.replace('.npy', '.png'))
            rgb = imageio.imread(rgb_file) / 255.0  # (376, 1408, 3)

            """Todo: whether add mask erosion or dilation"""
            # cosistency_mask = np.logical_and(cosistency_mask,errosion_mask)
            if self.filter_sky:
                instance_file = os.path.join(self.dir_name, 'semantic', 'instance', file_name.replace('.npy', '.png'))
                instance = cv2.imread(instance_file, -1)
                erosion = np.ones_like(rgb)
                erosion[instance==10] = (0, 0, 0)
                erosion[instance!=10] = (255, 255, 255)
                ## add erosion for sky binary mask
                kernel = np.ones((30, 30), np.uint8)
                erosion = cv2.erode(erosion, kernel, iterations=1)
                mask = np.all(erosion != [0, 0, 0], axis=2)
                fina_mask = np.logical_and(cosistency_mask[i], mask)
            else:
                 fina_mask = cosistency_mask[i]

            kept = np.argwhere(fina_mask)

            depth = depth[kept[:, 0], kept[:, 1]]
            rgb = rgb[kept[:, 0], kept[:, 1]]
            c2w = self.c2w[i]
            K = self.intri[i]
                        
            x = np.arange(0, self.W)  # generate pixel coordinates
            y = np.arange(0, self.H)
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(self.H, self.W, 2)

            x = (pixels[kept[:, 0], kept[:, 1]][:, 0] - K[0,2]) * depth / K[0,0]
            y = (pixels[kept[:, 0], kept[:, 1]][:, 1] - K[1,2]) * depth / K[1,1]
            z = depth
            coordinates = np.stack([x, y, z], axis=1)
            coordinates = np.column_stack((coordinates.reshape(-1, 3), np.ones(len(coordinates.reshape(-1, 3)))))
            
            worlds = np.dot(c2w, coordinates.T).T
            worlds = worlds[:, :3]


            color_pointclouds.append(np.concatenate([worlds, rgb.reshape(-1, 3)], axis=-1))

        point_clouds = np.concatenate(color_pointclouds, axis=0).reshape(-1, 6)
        return point_clouds


    
    def depth_cosistency_check(self,depth_files):
        depth_masks = []
        print("Depth Check!")
        for i, file_name in enumerate(depth_files):
            depth_file = os.path.join(self.depth_dir, file_name)
            depth = np.load(depth_file)

            """ We assume the first depth frame is correct """
            if i == 0:
                last_depth = deepcopy(depth)
                # last_file_name = deepcopy(file_name)
                depth_masks.append(np.ones((self.H,self.W)))
                continue

            c2w = self.c2w[i]
            last_c2w = self.c2w[i-1]
            K = self.intri[i]

            ## unproject pointcloud
            x = np.arange(0, depth.shape[1])  # generate pixel coordinates
            y = np.arange(0, depth.shape[0])
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

            # unproject depth to pointcloud
            x = (pixels[..., 0] - K[0,2]) * depth.reshape(-1) / K[0,0]
            y = (pixels[..., 1] - K[1,2]) * depth.reshape(-1) / K[0,0]
            z = depth.reshape(-1)
            coordinates = np.stack([x, y, z], axis=1)

            depth_mask = self.depth_projection_check(coordinates=coordinates,pixels=pixels,last_c2w=last_c2w,c2w=c2w,last_depth=last_depth,depth=depth,K=K)
            depth_masks.append(depth_mask)

            ## update status
            last_depth = deepcopy(depth)
        
        return depth_masks
    
    def depth_projection_check(self,coordinates, pixels, last_c2w, c2w, last_depth, depth,K):
        H,W = last_depth.shape[:2]
        cx,cy = K[0,2], K[1,2]
        fx, fy = K[0,0], K[1,1]

        trans_mat = np.dot(np.linalg.inv(last_c2w), c2w)
        coordinates_homo = np.column_stack((coordinates.reshape(-1, 3), np.ones(len(coordinates.reshape(-1, 3)))))
        last_coordinates = np.dot(trans_mat, coordinates_homo.T).T
        last_x = (fx * last_coordinates[:, 0] + cx * last_coordinates[:, 2]) / last_coordinates[:, 2]
        last_y = (fy * last_coordinates[:, 1] + cy * last_coordinates[:, 2]) / last_coordinates[:, 2]
        last_pixels = np.vstack((last_x, last_y)).T.reshape(-1, 2).astype(np.int32)
        
        pixels[:, [0, 1]] = pixels[:, [1, 0]]
        last_pixels[:, [0, 1]] = last_pixels[:, [1, 0]]
        
        depth_mask = np.ones(depth.shape[0]*depth.shape[1])

        """Reprojection location must in image plane [0,H]  [0,W] """
        valid_mask_00 = np.logical_and((last_pixels[:, 0] < H), (last_pixels[:, 1] < W))
        valid_mask_01 = np.logical_and((last_pixels[:, 0] > 0), (last_pixels[:, 1] > 0))
        valid_mask = np.logical_and(valid_mask_00, valid_mask_01)

        depth_diff = np.abs(depth[pixels[valid_mask, 0], pixels[valid_mask, 1]] - last_depth[last_pixels[valid_mask, 0], last_pixels[valid_mask, 1]])
        depth_mask[valid_mask] = np.where(depth_diff < depth_diff.mean(), 1, 0)
        depth_mask = depth_mask.reshape(*depth.shape)

        return depth_mask




