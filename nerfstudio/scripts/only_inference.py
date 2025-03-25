
"""
eval.py
"""
from __future__ import annotations


from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

import mediapy as media
import numpy as np
import torch
from tqdm import tqdm
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import cv2
from torchvision.transforms.functional import resize as gpu_resize

CONSOLE = Console(width=120)

class RenderDatasets():
    """Load a checkpoint, render the trainset and testset rgb,normal,depth, and save to the picture"""
    def __init__(self,parser_path):
        self.load_config = Path(parser_path.config)

        exp_method = str(self.load_config).split('/')[-3]
        descriptor = str(self.load_config).split('/')[-2]
        if exp_method == 'nerfacto':
            self.rendered_output_names = ['rgb', 'depth',"semantics"]
        elif exp_method =='neuralsplat':
            self.rendered_output_names = ['rgb', 'depth','accumulation']
        else:
            self.rendered_output_names = ['rgb', 'depth', 'accumulation']
        root_dir = Path("Inferences_dir")
        self.root_dir = root_dir / Path('infer_' + descriptor)
        if self.root_dir.is_dir():
            os.system(f"rm -rf {self.root_dir}")
        self.task = parser_path.task
        self.is_leaderboard = parser_path.is_leaderboard
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity().cuda()
        self.downscale = parser_path.downscale

    def draw_colorbar_depth(self,render_depth):
        for i in range(len(render_depth)):
            pred_depth = render_depth[i].squeeze(2)
            pred_depth = pred_depth.clip(0,20)
            print(f"Predecited Depth Max:{pred_depth.max()}  clip max = 20 ")
            plt.close('all')
            ax = plt.subplot()
            sc = ax.imshow((pred_depth), cmap='turbo')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(sc, cax=cax)
            plt.savefig(os.path.join(str(self.root_dir)+f'/{self.task}_{i:02d}_depth.png'))
        CONSOLE.print(f"[bold blue] Store image to {self.root_dir}")

    def generate_MSE_map(self,redner_img,gt_img,index):
        mse = np.mean((redner_img - gt_img) ** 2,axis=-1)
        plt.close('all')
        plt.figure(figsize=(15, 5))  ## figure 的宽度:1500 height: 500
        ax = plt.subplot()
        sc = ax.imshow((mse), cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)
        plt.savefig(os.path.join(str(self.root_dir)+"/error_map"+ f'/{self.task}_{index:02d}_mse.png'), bbox_inches='tight')
        return


    def main(self):
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        
        trainDataCache = pipeline.datamanager.train_dataset
        testDatasetCache = pipeline.datamanager.eval_dataset
        os.makedirs(self.root_dir / "error_map", exist_ok=True)
        os.makedirs(self.root_dir / "gt_rgb", exist_ok=True)

        ## Not add appearance embedding
        if self.task == 'trainset':
            DataCache = trainDataCache
        elif self.task == 'testset':
            DataCache = testDatasetCache
        else:
             raise ValueError("Task Input is trainset or testset")

        num_images = len(DataCache.cameras) #type: ignore
        config.print_to_terminal()
        'Read the image and save in target directory'
        os.makedirs(self.root_dir / "render_rgb",exist_ok=True)

        CONSOLE.print(f"[bold yellow]Rendering {num_images} Images")
        

        render_image = []
        render_depth = []
        render_accumulation = []
        gt_image = []

        for camera_idx in tqdm(range(num_images)):
            with torch.no_grad():
                if self.task == 'testset':
                    camera, batch = pipeline.datamanager.next_fixed_eval_image(camera_idx)
                    gt_image.append(batch['target']['image'])
                else:
                    camera, batch = pipeline.datamanager.next_fixed_train(camera_idx)
                    gt_image.append(batch['target']['image'])
                
                outputs = pipeline.model.get_outputs_for_camera(camera,batch)
            for rendered_output_name in self.rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}",
                                    justify="center")
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                if rendered_output_name == 'rgb':
                    render_image.append(output_image)
                elif rendered_output_name == 'depth':
                    render_depth.append(output_image)
                elif rendered_output_name == 'accumulation':
                    render_accumulation.append(output_image)

        CONSOLE.print("[bold green]Rendering Images Finished")

        ''' Output rgb depth and normal image'''
        sum_psnr = 0
        sum_lpips = 0
        sum_ssim = 0
        for i in range(num_images):
            image = gt_image[i].squeeze(0)
            if self.downscale != 1:
                H_ = int(image.shape[0] // self.downscale)
                W_ = int(image.shape[1] // self.downscale)
                image = gpu_resize(image.movedim(-1, 0),(H_, W_)).movedim(0, -1) # type: ignore

            combine_image = np.concatenate([render_image[i],image.detach().cpu().numpy()],axis=0)
            media.write_image(self.root_dir / f'{self.task}_{i:02d}_redner_rgb.png',combine_image )
            media.write_image(self.root_dir / "render_rgb" / f'{self.task}_{i:02d}_pred.png', render_image[i])
            media.write_image(self.root_dir/"gt_rgb" / f'{self.task}_{i:02d}_gtrgb.png', (image.detach().cpu().numpy()))
         
            depth_map = colormaps.apply_depth_colormap(depth=torch.from_numpy(render_depth[i]),accumulation=torch.from_numpy(render_accumulation[i]))
            media.write_image(self.root_dir /  f'{self.task}_{i:02d}_depth.png', depth_map)

            self.generate_MSE_map(image.detach().cpu().numpy(),render_image[i],i)
            psnr = -10. * np.log10(np.mean(np.square(image.detach().cpu().numpy() - render_image[i])))
            lpips = self.lpips(image.unsqueeze(0).permute(0,3,1,2),torch.from_numpy(render_image[i]).unsqueeze(0).permute(0,3,1,2).cuda())
            SSIM = self.ssim(image.unsqueeze(0).permute(0,3,1,2),torch.from_numpy(render_image[i]).unsqueeze(0).permute(0,3,1,2).cuda())

            sum_psnr += psnr
            sum_lpips += lpips
            sum_ssim += SSIM
            print("{} Mode image {} PSNR:{} LPIPS: {} SSIM: {}".format(self.task,i,psnr,lpips, SSIM))


        CONSOLE.print(f"[bold green]Average PSNR:{sum_psnr/ num_images}",justify="center")
        CONSOLE.print(f"[bold green]Average LPIPS: {sum_lpips / num_images }",justify="center")
        CONSOLE.print(f"[bold green]Average ssim:{sum_ssim/ num_images}",justify="center")
        CONSOLE.print(f"[bold red] Saved in {self.root_dir}")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='testset or trainset')
    parser.add_argument('--config',type=str,help='Config Path')
    parser.add_argument('--is_leaderboard',action='store_true')
    parser.add_argument("--downscale",type=int,help= "downscale the image",default=1)
    config = parser.parse_args()

    RenderDatasets(config).main()

