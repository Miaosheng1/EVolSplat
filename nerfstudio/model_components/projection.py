# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class Projector():
    def __init__(self):
        print("Init the Projector in OpenGL system")

    
    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras,train_intrinsics):
        '''
        project 3D points into cameras
        :param xyz: [..., 3]  Opencv
        :param train_cameras: [n_views, 4, 4]  OpenGL
        :param camera intrinsics: [n_views, 4, 4]
        :return: pixel locations [..., 2], mask [...]
        '''
        original_shape = xyz.shape[:1]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_cameras = train_cameras * torch.tensor([1, -1, -1, 1],device="cuda")
        train_poses = train_cameras.reshape(-1, 4, 4)  # [n_views, 4, 4]

        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera

        depth = projections[..., 2].reshape((num_views, ) + original_shape)
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape),\
               depth
    
    ## compute the projection of the query points to model the Background
    def compute(self,  xyz, train_imgs, train_cameras, train_intrinsics,cam_idx=0):
        '''
        :param xyz: [n_samples, 3]
        :param source_imgs: [ n_views, c, h, w]
        :param source_cameras: [ n_views, 4, 4], in OpnecGL
        :param source_intrinsics: [ n_views, 4, 4]
        :return: rgb_feat_sampled: [n_samples,n_views,c],
                 mask: [n_samples,n_views,1]
        '''
    
        xyz = xyz.detach()
        h, w = train_imgs.shape[2:]

        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front, _ = self.compute_projections(xyz, train_cameras,train_intrinsics.clone())
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_points, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(dim=1) # [n_views, 1, n_points, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=False)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1).squeeze(dim=0)  # [n_points, n_views, 3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1, 0)[..., None]   # [n_rays, n_samples, n_views, 1]
        rgb = rgb_sampled.masked_fill(mask==0, 0)

        projection_mask = mask[..., :].sum(dim=1) > 0
        return rgb[projection_mask.squeeze()], projection_mask.squeeze()
    

    def sample_within_window(self,  xyz, train_imgs, train_cameras, train_intrinsics, source_depth=None, local_radius = 2, depth_delta=0.2):
        '''
        :param xyz: [n_samples, 3]
        :param source_imgs: [ n_views, c, h, w]
        :param source_cameras: [ n_views, 4, 4], in OpnecGL
        :param source_intrinsics: [ n_views, 4, 4]
        :param source_depth: [ n_views , h, w] for occlusion-aware IBR
        :return: rgb_feat_sampled: [n_samples,n_views,c],
                 mask: [n_samples,n_views,1]
        '''
        n_views, _ ,_ = train_cameras.shape
        n_samples = xyz.shape[0]
        
        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1
        window_grid = self.generate_window_grid(-local_radius, local_radius,
                                                -local_radius, local_radius,
                                                local_h, local_w, device=xyz.device)  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(n_views, 1, 1)

        xyz = xyz.detach()
        h, w = train_imgs.shape[2:]

        # sample within the window size
        pixel_locations, mask_in_front, project_depth = self.compute_projections(xyz, train_cameras,train_intrinsics.clone())

        ## Occlusion-Aware check for IBR:
        if source_depth is not None:
            source_depth = source_depth.unsqueeze(-1).permute(0, 3, 1, 2).cuda()
            depths_sampled = F.grid_sample(source_depth, self.normalize(pixel_locations, h, w).unsqueeze(dim=1), align_corners=False)
            depths_sampled = depths_sampled.squeeze()
            retrived_depth = depths_sampled.masked_fill(mask_in_front==0, 0)
            projected_depth = project_depth*mask_in_front

            """Use depth priors to distinguish the Occlusion Region"""
            visibility_map = projected_depth - retrived_depth
            visibility_map = visibility_map.unsqueeze(-1).repeat(1,1, local_h*local_w).reshape(n_views,n_samples,-1)
        else:
            visibility_map = torch.ones_like(project_depth)

        pixel_locations = pixel_locations.unsqueeze(dim=2) + window_grid.unsqueeze(dim=1)
        pixel_locations = pixel_locations.reshape(n_views,-1,2)  ## [N_view, N_points,2]

        ## boardcasting the mask
        mask_in_front = mask_in_front.unsqueeze(-1).repeat(1,1, local_h*local_w).reshape(n_views,-1)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_points, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(dim=1) # [n_views, 1, n_points, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=False)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1).squeeze(dim=0)  # [n_points, n_views, 3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front ).float().permute(1, 0)[..., None]  
        rgb = rgb_sampled.masked_fill(mask==0, 0)

        return rgb.reshape(n_samples,n_views,local_w*local_h,3), \
                mask.reshape(n_samples,n_views,local_w*local_h),\
                visibility_map.permute(1,0,2).unsqueeze(-1)
    
    def generate_window_grid(self, h_min, h_max, w_min, w_max, len_h, len_w, device=None):
        assert device is not None

        x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                            torch.linspace(h_min, h_max, len_h, device=device)],
                            )
        grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

        return grid










