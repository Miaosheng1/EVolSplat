import numpy as np
import os
import imageio.v2 as imageio
import json
import cv2
import open3d as o3d
from typing import Literal

from copy import deepcopy

X_MIN, X_MAX = -16, 16
Y_MIN, Y_MAX = -9, 3.8
Z_MIN, Z_MAX = -30, 30

class PCDGenerator():
    ## "Drop25" discards the last frame out of every 4 frames
    sparsity: Literal['Drop90','Drop50',"Drop80","Drop25","full"] = "full"
    use_bbx: bool = True
    """Accumulate the number of pointcloud frames"""

    def __init__(self,spars='full',save_dir="Drop50",frame_start=0, filer_sky=True,depth_cosistency=True) -> None:
        self.sparsity = spars
        self.save_dir = save_dir
        self.cam2world_dict_00 = None
        self.dir_name = None
        self.frame_start = frame_start
        self.filter_sky = filer_sky
        self.depth_cosistency = depth_cosistency


    def set_image_info(self,info):
        K, H, W = info[0], info[1], info[2]
        self.H = H
        self.W = W
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx, self.cy = K[0,2], K[1,2]
    
    def get_bbx(self):
        return np.array([X_MIN, Y_MIN, Z_MIN]), np.array([X_MAX, Y_MAX, Z_MAX])


    def crop_pointcloud(self,bbx_min, bbx_max, points, color):
        mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])

        return points[mask], color[mask]
    
    def crop_pointcloud_outside(self,bbx_min, bbx_max, points, color):
        mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])

        return points[~mask], color[~mask]

    def forward(self, dir_name, cam2world_dict_00, cam2world_dict_01, down_scale=2):

        self.dir_name = dir_name
        self.depth_dir = os.path.join(self.dir_name, 'depth')
        self.c2ws = []

        depth_files = []
        selected_index = []
        
        """ Process differnt Sparsity level KITTI-360 """
        for i, file_name in enumerate(sorted(os.listdir(self.depth_dir))):
            if self.sparsity == "Drop50":
               if i % 4 == 2 or i % 4 == 3:
                     continue
            elif self.sparsity == 'Drop80':
                if i % 10 != 0 or i % 10 != 1:
                    continue 
            elif self.sparsity == 'Drop25':
                if i % 4 == 2:
                    continue 
            elif self.sparsity == 'Drop90':
                if i % 10 != 0:
                    continue
        
            depth_files.append(file_name)
            selected_index.append(i)

        print(f"{self.sparsity} :ALL frames: {len(depth_files)}:,{selected_index} \n")
        self.extract_c2w(depth_files=depth_files,cam2world_dict_00=cam2world_dict_00,cam2world_dict_01=cam2world_dict_01)

      
        if self.depth_cosistency:
            cosistency_mask = self.depth_cosistency_check(depth_files=depth_files)
        else:
            cosistency_mask = [np.ones((self.H,self.W)) for _ in range(len(depth_files))]

        accmulat_pointcloud, single_pointcloud = self.accumulat_pcd(depth_files=depth_files,cosistency_mask=cosistency_mask,down_scale=down_scale)

        """ Output and Save the .ply pointcloud """
        os.makedirs(os.path.join(self.dir_name, self.save_dir),exist_ok=True)

        points = accmulat_pointcloud[:, :3]
        colors = accmulat_pointcloud[:, 3:]
        with open(os.path.join(self.dir_name, 'transforms.json'), 'r') as f:
            transform = json.load(f)

        ## Transform into center normalize coordinates
        w2c = np.array(transform['inv_pose']).astype(np.float32)
        pts_camera = np.hstack((points, np.ones((points.shape[0], 1))))
        pts_camera = np.dot(pts_camera, np.transpose(w2c))   # Transform to camera coordinate system

        "Even Though we use the bounding box, we still comprise the distant background pointcloud outside the bbx"
        if self.use_bbx:
            bbx_min, bbx_max = self.get_bbx()
            print(f"BBX Range: {bbx_min},{bbx_max} \n")
            pts_camera, colors = self.crop_pointcloud(bbx_min, bbx_max, pts_camera, colors)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pts_camera[:, :3])
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
            point_cloud = point_cloud.select_by_index(ind)
            point_cloud = point_cloud.uniform_down_sample(every_k_points=5)
      
        else:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pts_camera[:, :3])
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

        ## Filter the noisy pointcloud and downsample the pcd in 3D space

        o3d.io.write_point_cloud(os.path.join(self.dir_name, self.save_dir, f'{self.frame_start}.ply'), point_cloud)
        print(f"Save the pointcloud in {os.path.join(self.dir_name, self.save_dir)} !")

    
    def extract_c2w(self, depth_files, cam2world_dict_00, cam2world_dict_01):
        for depth_file in depth_files:
            camera_id = int(depth_file.split('_')[0])
            suffix = depth_file.split('_')[1][:2]

            if suffix == '00':
                self.c2ws.append(cam2world_dict_00[camera_id])
            elif suffix == '01':
                self.c2ws.append(cam2world_dict_01[camera_id])
            else:
                raise ValueError("Invalid suffix")



    """The accumulated pointcloud locate in the OpenCV system """
    def accumulat_pcd(self,depth_files,cosistency_mask,down_scale:int = 2):
        color_pointclouds = []
        num_images = len(depth_files)
        if down_scale != 1:
            downscale_mask =  np.zeros_like(cosistency_mask[0])
            downscale_mask[::down_scale,::down_scale] = 1

        for i, file_name in enumerate(depth_files):
            depth_file = os.path.join(self.depth_dir, file_name)
            depth = np.load(depth_file)  # (376, 1408)
            rgb_file = os.path.join(self.dir_name, file_name.replace('.npy', '.png'))
            rgb = imageio.imread(rgb_file) / 255.0  # (376, 1408, 3)

            """Todo: whether add mask erosion or dilation"""
            if self.filter_sky:
                instance_file = os.path.join(self.dir_name, 'semantic', 'instance', file_name.replace('.npy', '.png'))
                instance = cv2.imread(instance_file, -1)
                erosion = np.ones_like(rgb)
                erosion[instance==10] = (0, 0, 0)
                erosion[instance!=10] = (255, 255, 255)
                ## add erosion for sky binary mask
                kernel = np.ones((1, 1), np.uint8)
                erosion = cv2.erode(erosion, kernel, iterations=1)
                mask = np.all(erosion != [0, 0, 0], axis=2)
                fina_mask = np.logical_and(cosistency_mask[i], mask)
            else:
                 fina_mask = cosistency_mask[i]

            ## downsample the 2D image to generate the pointcloud
            if down_scale != 1:
                fina_mask = np.logical_and(downscale_mask, fina_mask)

            kept = np.argwhere(fina_mask)

            depth = depth[kept[:, 0], kept[:, 1]]
            rgb = rgb[kept[:, 0], kept[:, 1]]
            c2w = self.c2ws[i]
                        
            x = np.arange(0, self.W)  # generate pixel coordinates
            y = np.arange(0, self.H)
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(376, 1408, 2)

            x = (pixels[kept[:, 0], kept[:, 1]][:, 0] - self.cx) * depth / self.fx
            y = (pixels[kept[:, 0], kept[:, 1]][:, 1] - self.cy) * depth / self.fy
            z = depth
            coordinates = np.stack([x, y, z], axis=1)
            coordinates = np.column_stack((coordinates.reshape(-1, 3), np.ones(len(coordinates.reshape(-1, 3)))))
            
            worlds = np.dot(c2w, coordinates.T).T
            worlds = worlds[:, :3]
            color_pointclouds.append(np.concatenate([worlds, rgb.reshape(-1, 3)], axis=-1))

        point_clouds = np.concatenate(color_pointclouds, axis=0).reshape(-1, 6)
        bg_pcd = np.concatenate([color_pointclouds[0],color_pointclouds[num_images//2],color_pointclouds[num_images-10]], axis=0).reshape(-1, 6)
        return point_clouds, bg_pcd


    
    def depth_cosistency_check(self,depth_files):
        depth_masks = []
        
        for i, file_name in enumerate(depth_files):
            depth_file = os.path.join(self.depth_dir, file_name)
            depth = np.load(depth_file)

            """ We assume the first depth frame is correct """
            if i == 0:
                last_depth = deepcopy(depth)
                # last_file_name = deepcopy(file_name)
                depth_masks.append(np.ones((self.H,self.W)))
                continue

            c2w = self.c2ws[i]
            last_c2w = self.c2ws[i-1]

            ## unproject pointcloud
            x = np.arange(0, depth.shape[1])  # generate pixel coordinates
            y = np.arange(0, depth.shape[0])
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

            # unproject depth to pointcloud
            x = (pixels[..., 0] - self.cx) * depth.reshape(-1) / self.fx
            y = (pixels[..., 1] - self.cy) * depth.reshape(-1) / self.fy
            z = depth.reshape(-1)
            coordinates = np.stack([x, y, z], axis=1)

            depth_mask = self.depth_projection_check(coordinates=coordinates,pixels=pixels,last_c2w=last_c2w,c2w=c2w,last_depth=last_depth,depth=depth)
            depth_masks.append(depth_mask)

            ## update status
            last_depth = deepcopy(depth)
            # last_file_name = deepcopy(file_name)
        
        return depth_masks
    
    def depth_projection_check(self,coordinates, pixels, last_c2w, c2w, last_depth, depth):
        H,W = last_depth.shape[:2]

        trans_mat = np.dot(np.linalg.inv(last_c2w), c2w)
        coordinates_homo = np.column_stack((coordinates.reshape(-1, 3), np.ones(len(coordinates.reshape(-1, 3)))))
        last_coordinates = np.dot(trans_mat, coordinates_homo.T).T
        last_x = (self.fx * last_coordinates[:, 0] + self.cx * last_coordinates[:, 2]) / last_coordinates[:, 2]
        last_y = (self.fy * last_coordinates[:, 1] + self.cy * last_coordinates[:, 2]) / last_coordinates[:, 2]
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




