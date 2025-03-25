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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
import os
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_manner
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE
from omegaconf import OmegaConf
from tqdm import tqdm
MAX_AUTO_RESOLUTION = 1600


@dataclass
class EvolSplatDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: NeuralSplat)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    eval_mode: Literal["manner", "filename", "interval", "all"] = "manner"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    mask_color: Optional[Tuple[float, float, float]] = None
    """Replace the unknown pixels with this color. Relevant if you have a mask but still sample everywhere."""
    load_3D_points: bool = False
    """Whether to load the 3D points from the colmap reconstruction."""
    pcd_ration: int = 1
    """ the downscale ration of input pointcloud """
    include_depth: bool = True
    """whether or not to include loading of Metric Depth"""
    num_scenes: int = 180
    """Number of Pretrain Scenes"""

@dataclass
class NeuralSplat(DataParser):
    """Nerfstudio DatasetParser"""

    config: EvolSplatDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        train_path = sorted(os.listdir(Path(f"{self.config.data}")))
        assert self.config.num_scenes <= len(train_path)
        train_path = train_path[:self.config.num_scenes]
        self.config.num_scenes = len(train_path)
        CONSOLE.log("[yellow] Load Scene: ", self.config.num_scenes)

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []
        seed_point = []
        
        for i in tqdm(range(self.config.num_scenes)):
            data_dir =  self.config.data / Path(train_path[i])
            meta = load_from_json(data_dir / "transforms.json")
         
            fx_fixed = "fl_x" in meta
            fy_fixed = "fl_y" in meta
            cx_fixed = "cx" in meta
            cy_fixed = "cy" in meta
            height_fixed = "h" in meta
            width_fixed = "w" in meta
            distort_fixed = False
         

            # sort the frames by fname
            fnames = []
            for frame in meta["frames"]:
                filepath = Path(frame["file_path"])
                fname = self._get_fname(filepath, data_dir)
                fnames.append(fname)
            inds = np.argsort(fnames)
            frames = [meta["frames"][ind] for ind in inds]

            ## Read the input pointcloud; We use the Drop50% Pointcloud for pretraining
            if self.config.load_3D_points and split == "train":
                if "ply_file_path_Drop50" in meta:
                    ply_file_path = data_dir / meta["ply_file_path_Drop50"]
                else:
                    CONSOLE.print(
                        "[bold yellow]Warning: load_3D_points set to true but no point cloud found. splatfacto will use random point cloud initialization."
                    )
                    ply_file_path = None

                if ply_file_path:
                    sparse_points = self._load_3D_points(ply_file_path, ratio=self.config.pcd_ration)
                    if sparse_points is not None:
                        seed_point.append(sparse_points)
            for idx,frame in enumerate(frames):
                filepath = Path(frame["file_path"])
                fname = self._get_fname(filepath, data_dir)

                if not fx_fixed:
                    assert "fl_x" in frame, "fx not specified in frame"
                    fx.append(float(frame["fl_x"]))
                if not fy_fixed:
                    assert "fl_y" in frame, "fy not specified in frame"
                    fy.append(float(frame["fl_y"]))
                if not cx_fixed:
                    assert "cx" in frame, "cx not specified in frame"
                    cx.append(float(frame["cx"]))
                if not cy_fixed:
                    assert "cy" in frame, "cy not specified in frame"
                    cy.append(float(frame["cy"]))
                if not height_fixed:
                    assert "h" in frame, "height not specified in frame"
                    height.append(int(frame["h"]))
                if not width_fixed:
                    assert "w" in frame, "width not specified in frame"
                    width.append(int(frame["w"]))
                if not distort_fixed:
                    distort.append(
                        torch.tensor(frame["distortion_params"], dtype=torch.float32)
                        if "distortion_params" in frame
                        else camera_utils.get_distortion_params(
                            k1=float(frame["k1"]) if "k1" in frame else 0.0,
                            k2=float(frame["k2"]) if "k2" in frame else 0.0,
                            k3=float(frame["k3"]) if "k3" in frame else 0.0,
                            k4=float(frame["k4"]) if "k4" in frame else 0.0,
                            p1=float(frame["p1"]) if "p1" in frame else 0.0,
                            p2=float(frame["p2"]) if "p2" in frame else 0.0,
                        )
                    )

                image_filenames.append(fname)
                poses.append(np.array(frame["transform_matrix"]))

            if self.config.include_depth:
                depth_filepath = data_dir/Path('depth')
                depth_filenames += [os.path.join(depth_filepath, f) for f in sorted(os.listdir(depth_filepath))]
                

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        if self.config.eval_mode == "manner":
            num_images = len(image_filenames)
            i_all = np.arange(num_images)
            i_eval = sorted(np.concatenate([np.arange(10,num_images,50),np.arange(30,num_images,50)]))
            i_train = np.setdiff1d(i_all, i_eval)
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]


        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata={},
        )
 
        # reinitialize metadata for dataparser_outputs
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=1.0,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "mask_color": self.config.mask_color,
                "input_pnt": seed_point,
            },
        )
        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path, ratio: int = 3):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))
        # if no points found don't read in an initial point cloud
        if len(pcd.points) == 0:
            return None
        points3D = np.asarray(pcd.points, dtype=np.float32)[::ratio,:]

        points3D = torch.from_numpy(points3D)
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))[::ratio,:]

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }

        return out

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file."""
        return data_dir / filepath
    

    def crop_pointcloud(self,bbx_min, bbx_max, points, color):
        mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])

        return points[mask], color[mask]
