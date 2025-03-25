# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases
from pytorch_msssim import SSIM
from torch.nn import Parameter
from torch import Tensor, nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import colormaps

from omegaconf import OmegaConf
from einops import rearrange
from nerfstudio.model_components.projection import Projector
from nerfstudio.field_components.mlp import MLP
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch.nn.functional as F
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.model_components.sparse_conv import sparse_to_dense_volume, SparseCostRegNet, construct_sparse_tensor
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.fields.initial_BgSphere import GaussianBGInitializer
from tqdm import tqdm


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class EvolSplatModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""
    _target: Type = field(default_factory=lambda: EvolSplatModel)
    validate_every: int = 8000
    """period of steps where gaussians are culled and densified"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    enabale_appearance_embedding: bool = False
    """whether enable the appearance embedding"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    entropy_loss: float = 0.1
    """weight of Entropy loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 1
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    """Config of the camera optimizer to use"""
    freeze_volume: bool = False


class EvolSplatModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: EvolSplatModelConfig

    def __init__(
        self,
        *args,
        seed_points: List,
        **kwargs,
    ):
        self.seed_points = seed_points
        self.num_scenes = len(seed_points) # type: ignore
        super().__init__(*args, **kwargs)

    def populate_modules(self, opts=None):

        ## Important: input the 3D point of the scene. All scenes data should be stroed as List as the unsame point number
        self.means = [] # type: ignore
        self.anchor_feats = []
        self.scales = [] # type: ignore
        self.offset = [] # type: ignore
        if self.seed_points is not None:
            for i in tqdm(range(self.num_scenes)):
                means = self.seed_points[i]['points3D_xyz']
                anchors_feat =   self.seed_points[i]['points3D_rgb'] / 255
                offsets = torch.zeros_like(means)
                distances, _ = self.k_nearest_sklearn(means.data, 3)
                distances = torch.from_numpy(distances)
                avg_dist = distances.mean(dim=-1, keepdim=True)
                scales = torch.log(avg_dist.repeat(1, 3))
                ## stack the parameters into list
                self.means.append(means)
                self.anchor_feats.append(anchors_feat)
                self.scales.append(scales)
                self.offset.append(offsets)
       
       

        ## load mannul param:
        assert opts is not None 
        self.local_radius = getattr(opts.model, 'local_radius', 1)
        self.sparseConv_outdim = opts.model.sparseConv_outdim
        self.offset_max = opts.model.offset_max 
        self.num_neibours = opts.model.num_neighbour_select 
        self.bbx_min = torch.tensor(opts.Boundingbox_min).float()
        self.bbx_max = torch.tensor(opts.Boundingbox_max).float()
        
        
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        ## config the projecter
        self.projector = Projector()

         ## construct the sparse tensor
        self.sparse_conv = SparseCostRegNet(d_in=3, d_out=self.sparseConv_outdim).cuda()
        self.voxel_size = opts.encoder.voxel_size
        
        self.feature_dim_out = 3*num_sh_bases(self.config.sh_degree)

        self.feature_dim_in = 4*self.num_neibours*(2*self.local_radius+1)**2
       
        if self.config.enabale_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_train_data, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None
        

        ## gaussian appearance MLP, predict the SH coefficients
        self.gaussion_decoder = MLP(
                in_dim= self.feature_dim_in+4,
                num_layers=3,
                layer_width=128,
                out_dim=self.feature_dim_out,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="torch",
            )
        
        self.mlp_conv = MLP(
                in_dim= self.sparseConv_outdim+4,
                num_layers=2,
                layer_width=64,
                out_dim=3+4,
                activation=nn.Tanh(),
                out_activation=None,
                implementation="torch",
            )
        
        self.mlp_opacity = MLP(
                in_dim=self.sparseConv_outdim+4,
                num_layers=2,
                layer_width=64,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="torch",
            )
        
        self.mlp_offset = MLP(
                in_dim=self.sparseConv_outdim,
                num_layers=2,
                layer_width=64,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Tanh(),
                implementation="torch",
            )
        
      
        ## Background Model for sky & distant view
        if self.config.enable_collider:
            Res = getattr(opts.bg_model,"res", 700)
            Radius = getattr(opts.bg_model,"radius", 25)
            self.scene_center = np.array(getattr(opts.bg_model,"center", [0,3.8,5.6]))
            gs_sky_initlial = GaussianBGInitializer(resolution=Res, radius=Radius,center=self.scene_center)
            bg_pnt = gs_sky_initlial.build_model()
            bg_distances, _ = self.k_nearest_sklearn(torch.from_numpy(bg_pnt), 3)
            bg_distances = torch.from_numpy(bg_distances)
            avg_dist = bg_distances.mean(dim=-1, keepdim=True)
            self.bg_scales = []
            self.bg_pcd = []
            for i in tqdm(range(self.num_scenes)):
                bg_scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
                bg_pcd = torch.tensor(bg_pnt)
                self.bg_scales.append(bg_scales)
                self.bg_pcd.append(bg_pcd)

            self.bg_field = MLP(
                in_dim=9,
                num_layers=2,
                layer_width=64,
                out_dim=6,
                activation=nn.ReLU(),
                out_activation=nn.Tanh(),
                implementation="torch",
            )
        
        self.renderer_rgb = RGBRenderer(background_color='black')



    def load_state_dict(self, model_dict, strict=False):  # type: ignore
        super().load_state_dict(model_dict, strict= strict)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    " CallBack Function"
    def after_train(self,step: int):
        return

   
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
                update_every_num_iters=self.config.validate_every,
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = {}
        ## add mlp decoder parameters
        gps['gaussianDecoder'] = list(self.gaussion_decoder.parameters())
        gps['mlp_conv'] = list(self.mlp_conv.parameters())
        gps['mlp_opacity'] = list(self.mlp_opacity.parameters())
        gps['mlp_offset'] = list(self.mlp_offset.parameters())
        gps['sparse_conv'] = list(self.sparse_conv.parameters())
        gps['background_model'] = list(self.bg_field.parameters())
        return gps


    def _downscale_if_required(self, image):
        d = 1
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self,BG_pcd, source_images, source_extrinsics, intrinsics):
        ## add projection mask to decrease the computational complexity
        sampled_feat,proj_mask = self.projector.compute(xyz = BG_pcd.reshape(-1,3), 
                                        train_imgs = source_images.squeeze(0),                
                                        train_cameras = source_extrinsics,     
                                        train_intrinsics= intrinsics, 
                                        )
        
        background_feat = self.bg_field(sampled_feat.view(-1,9))
        background_rgb, background_scale_res = background_feat.split([3,3],dim=-1)
        return background_rgb,proj_mask, torch.tanh(background_scale_res)
    

    def get_outputs(self, camera: Cameras,batch) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

        scene_id = batch.get("scene_id", None)
        means = self.means[scene_id].cuda()
        scales = self.scales[scene_id].cuda()
        offset = self.offset[scene_id].cuda()
        anchors_feat = self.anchor_feats[scene_id].cuda()
        
        optimized_camera_to_world = camera.camera_to_worlds

        source_images = batch['source']['image']
        source_images = rearrange(source_images[None,...],"b v h w c -> b v c h w")
        source_extrinsics = batch['source']['extrinsics'] 
        target_image = batch['target']['image'].squeeze(0)

        ## Query 3D features
        if not self.config.freeze_volume:
            sparse_feat, self.vol_dim, self.valid_coords = construct_sparse_tensor(raw_coords=means.clone(),
                                                                                   feats=anchors_feat,
                                                                                   Bbx_max=self.bbx_max,
                                                                                   Bbx_min=self.bbx_min,
                                                                                   voxel_size=self.voxel_size,
                                                                                   ) 
            feat_3d = self.sparse_conv(sparse_feat)
            dense_volume = sparse_to_dense_volume(sparse_tensor=feat_3d,coords=self.valid_coords,vol_dim=self.vol_dim).unsqueeze(dim=0)
            self.dense_volume = rearrange(dense_volume, 'B H W D C -> B C H W D')
       

        
          ## Query 2D features
        sampled_feat,valid_mask,vis_map = self.projector.sample_within_window(xyz = means, 
                                        train_imgs = source_images.squeeze(0),  ## [N_view,c,h,w]              
                                        train_cameras = source_extrinsics,      ## [N_view,4,4]
                                        train_intrinsics= batch['source']['intrinsics'],  ## [N_view,4,4]
                                        source_depth = batch['source']['depth'],
                                        local_radius=self.local_radius,
                                        )  # [N_samples, N_views, C] 
        
        sampled_feat = torch.concat([sampled_feat,vis_map],dim=-1).reshape(-1,self.feature_dim_in)
        valid_mask = valid_mask.reshape(-1,self.feature_dim_in//4)

        
        projection_mask = valid_mask[..., :].sum(dim=1) > self.local_radius**2 + 1
        num_pointcs = projection_mask.sum()
        means_crop = means[projection_mask]
        sampled_color = sampled_feat[projection_mask]
        vailid_scales = scales[projection_mask]
        last_offset = offset[projection_mask]

        ## Trilinear the feature volume
        grid_coords = self.get_grid_coords(means_crop + last_offset)
        feat_3d = self.interpolate_features(grid_coords=grid_coords, feature_volume=self.dense_volume).permute(3, 4, 1, 0, 2).squeeze()

        ## Add the relative direction and distance
        with torch.no_grad():
            ob_view = means_crop - optimized_camera_to_world[0,:3,3]
            ob_dist = ob_view.norm(dim=1, keepdim=True)
            ob_view = ob_view / ob_dist

        if self.config.enabale_appearance_embedding:
            if self.training :
                camera_indicies = torch.ones(num_pointcs, dtype=torch.long, device=ob_dist.device) * camera.metadata['cam_idx'] #type: ignore
                embedded_appearance = self.embedding_appearance(camera_indicies) #type: ignore
            else: 
                test_id = torch.ones(num_pointcs, dtype=torch.long, device=ob_dist.device) * camera.metadata['cam_idx']      #type: ignore
                embedded_appearance = 0.5 * (self.embedding_appearance(test_id+2) + self.embedding_appearance(test_id - 2))  #type: ignore
        
              #[N,6]
            input_feature = torch.cat([sampled_color, ob_dist, ob_view,embedded_appearance], dim=-1).squeeze(dim=1)
        else:
            input_feature = torch.cat([sampled_color, ob_dist, ob_view], dim=-1).squeeze(dim=1)
      
        sh = self.gaussion_decoder(input_feature)
        features_dc_crop = sh[:,:3]
        features_rest_crop = sh[:,3:].reshape(num_pointcs,-1,3)
        
        ## Learn 3D scale, rotation and opacity parameters
        scale_input_feat = torch.cat([feat_3d, ob_dist, ob_view],dim=-1).squeeze(dim=1)
        scales_crop, quats_crop = self.mlp_conv(scale_input_feat).split([3,4],dim=-1)
        opacities_crop = self.mlp_opacity(scale_input_feat) 

        ## Optimize the 3D offset via MLP
        offset_crop = self.offset_max * self.mlp_offset(feat_3d)
        means_crop += offset_crop

        ## Update the latest offset for each 3DGS; only save the tensor without grad
        if self.training:
            self.offset[scene_id][projection_mask] = offset_crop.detach().cpu()  

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)


        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        viewmat = get_viewmat(optimized_camera_to_world)
        K = batch['target']['intrinsics'][...,:3,:3]
        H, W = target_image.shape[:2]
        self.last_size = (H, W)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"


        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop + vailid_scales),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,  ## set True for more memory efficient
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=self.config.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
        )

        alpha = alpha[:, ...][0]
        render_rgb = render[:, ..., :3].squeeze(0)

        ## Model Background. We place the hemisphere background at the approximate center of the scene
        ## According to the scene bounding box
        center_z = self.scene_center[2]
        bg_offset = optimized_camera_to_world[0,2,3] - center_z
        bg_pcd = self.bg_pcd[scene_id].cuda() 
        bg_pcd[:,2] += bg_offset
       
        bg_scale = self.bg_scales[scene_id].cuda()
        background_feat,proj_mask, background_scale_res = self._get_background_color(BG_pcd=bg_pcd,
                                                                                    source_images=source_images.squeeze(0),
                                                                                    source_extrinsics= source_extrinsics,
                                                                                    intrinsics= batch['source']['intrinsics'],
                                                                                    )
        num_bg_points = background_feat.shape[0]
        bg_opacity = torch.ones(num_bg_points, 1).cuda()
        bg_quat = torch.tensor([[1.0,0,0,0]]).repeat(num_bg_points,1).cuda()
        

        bg_render, _, _ = rasterization(
            means=bg_pcd[proj_mask],
            quats=bg_quat / bg_quat.norm(dim=-1, keepdim=True),
            scales=torch.exp(bg_scale)[proj_mask] + background_scale_res,
            opacities=bg_opacity.squeeze(-1),
            colors=background_feat,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,  ## set True for more memory efficient
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=None,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
        )
        background_rgb = bg_render[:, ..., :3].squeeze(0)
        
        
        rgb = render_rgb + (1 - alpha) * background_rgb
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None


        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": (1 - alpha) * background_rgb,  # type: ignore
        }  # type: ignore
    
    
    def interpolate_features(self, grid_coords, feature_volume):
        grid_coords = grid_coords[None, None, None, ...]
        feature = F.grid_sample(feature_volume,
                                grid_coords,
                                mode='bilinear',
                                align_corners=True,
                                )
        return feature
    
    def get_grid_coords(self, position_w, voxel_size=[0.1,0.1,0.1]):
        assert self.voxel_size == voxel_size[0]
        bounding_min = self.bbx_min
        pts = position_w - bounding_min.to(position_w)
        x_index = pts[..., 0] / voxel_size[0]
        y_index = pts[..., 1] / voxel_size[1]
        z_index = pts[..., 2] / voxel_size[2]
        """ Normalize the point coordinates to [-1,1]"""

        dhw = torch.stack([x_index, y_index, z_index], dim=1)

        # index = dhw.clone().long()
        dhw[..., 0] = dhw[..., 0] / self.vol_dim[0] * 2 - 1
        dhw[..., 1] = dhw[..., 1] / self.vol_dim[1] * 2 - 1
        dhw[..., 2] = dhw[..., 2] / self.vol_dim[2] * 2 - 1
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords
    

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return image.to(self.device)


    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = batch['target']["image"].squeeze(0)
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        
        gt_img = batch['target']["image"].squeeze(0)
        pred_img = outputs["rgb"]
       
        if self.step % 10 == 0:
            entorpy_loss =  self.config.entropy_loss * (
                            - outputs['accumulation'] * torch.log(outputs['accumulation'] + 1e-10)
                            - (1 - outputs['accumulation']) * torch.log(1 - outputs['accumulation'] + 1e-10)
                            ).mean()
        else:
            entorpy_loss = torch.tensor(0.0).to(self.device)


        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "entorpy_loss": entorpy_loss,
        }

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, batch=None, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        if self.collider is not None and batch.has_key('raybundle'):
            batch['raybundle'] = self.collider(batch['raybundle'])  # type: ignore 
        outs = self.get_outputs(camera.to(self.device),batch=batch)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = batch['target']["image"].squeeze(0)# type: ignore
        predicted_rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        bg_color = outputs['background'].squeeze(0)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth, "background":bg_color}

        return metrics_dict, images_dict
    
    @torch.no_grad()
    def init_volume(self, scene_id:int = 0):
        ## Foreground
        self.config.freeze_volume = True
        means = self.means[scene_id].cuda()
        anchors_feat = self.anchor_feats[scene_id].cuda()
        sparse_feat, self.vol_dim, self.valid_coords = construct_sparse_tensor(raw_coords=means.clone(),
                                                                               feats=anchors_feat,
                                                                                Bbx_max=self.bbx_max,
                                                                                Bbx_min=self.bbx_min,
                                                                                   ) 
        feat_3d = self.sparse_conv(sparse_feat) # type: ignore
        dense_volume = sparse_to_dense_volume(sparse_tensor=feat_3d,coords=self.valid_coords,vol_dim=self.vol_dim).unsqueeze(dim=0)
        self.dense_volume = rearrange(dense_volume, 'B H W D C -> B C H W D')

        ## Refine locations of3D Gaussian Primitives 
        grid_coords = self.get_grid_coords(means)
        feat_3d = self.interpolate_features(grid_coords=grid_coords, feature_volume=self.dense_volume).permute(3, 4, 1, 0, 2).squeeze()

        offset_crop = self.offset_max * self.mlp_offset(feat_3d)
        self.offset[scene_id] = offset_crop.detach().cpu()  
        CONSOLE.print(f"[bold green] Freeze the feature volume and perform feed-forward inference on a target scene.",justify="center")
        return
    

    "Output global Gaussian Splats for a novel scene"
    @torch.no_grad()
    def output_evosplat(self, ref_origin:Tensor = torch.tensor([0,0,0]), scene_id:int = 0):
        self.config.freeze_volume = True
        means = self.means[scene_id].cuda()
        anchors_feat = self.anchor_feats[scene_id].cuda()
        sparse_feat, self.vol_dim, self.valid_coords = construct_sparse_tensor(raw_coords=means.clone(),
                                                                               feats=anchors_feat,
                                                                                Bbx_max=self.bbx_max,
                                                                                Bbx_min=self.bbx_min,
                                                                                   ) 
        feat_3d = self.sparse_conv(sparse_feat) # type: ignore
        dense_volume = sparse_to_dense_volume(sparse_tensor=feat_3d,coords=self.valid_coords,vol_dim=self.vol_dim).unsqueeze(dim=0)
        self.dense_volume = rearrange(dense_volume, 'B H W D C -> B C H W D')

        ## Update 3D Gaussian Splatting locations
        grid_coords = self.get_grid_coords(means)
        feat_3d = self.interpolate_features(grid_coords=grid_coords, feature_volume=self.dense_volume).permute(3, 4, 1, 0, 2).squeeze()

        offset_crop = self.offset_max * self.mlp_offset(feat_3d)
   
        distances, _ = self.k_nearest_sklearn(means, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.log(avg_dist.repeat(1, 3)).cuda()
        CONSOLE.print(f"[bold blue]Export Gaussians relative to the specific frame ... \n")
        gs_means = means + offset_crop

        ob_view = gs_means - ref_origin
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        scale_input_feat = torch.cat([feat_3d, ob_dist, ob_view],dim=-1).squeeze(dim=1)
        scales_crop, quats = self.mlp_conv(scale_input_feat).split([3,4],dim=-1)
        opacities = self.mlp_opacity(scale_input_feat) 

        gs_scales = torch.exp(scales_crop + scales)
        gs_opa = torch.sigmoid(opacities).squeeze(-1)
        gs_rot = quats / quats.norm(dim=-1, keepdim=True)
        gs_color = anchors_feat

        return {
            "means": gs_means, 
            "opacity": gs_opa, 
            "scales": gs_scales,  
            "rot": gs_rot,  
            "colors":gs_color
        }  # type: ignore
    