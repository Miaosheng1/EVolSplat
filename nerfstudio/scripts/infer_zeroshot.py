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

#!/usr/bin/env python
"""Train a radiance field with nerfstudio.
For real captures, we recommend using the [bright_yellow]nerfacto[/bright_yellow] model.

Nerfstudio allows for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    ns-train -h
    ns-train nerfacto -h nerfstudio-data
    ns-train nerfacto nerfstudio-data -h

In each of these examples, the -h applies to the previous subcommand (ns-train, nerfacto, and nerfstudio-data).

In the first example, we get the help menu for the ns-train script.
In the second example, we get the help menu for the nerfacto model.
In the third example, we get the help menu for the nerfstudio-data dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    ns-train nerfacto -h nerfstudio-data

need to come directly after the nerfacto subcommand, since these optional arguments only belong to the nerfacto
subcommand:

    ns-train nerfacto {nerfacto optional args} nerfstudio-data
"""

from __future__ import annotations

from pathlib import Path
import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
from tqdm import tqdm
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE
import os
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure
from nerfstudio.utils import colormaps
import mediapy as media
from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
import wandb
import moviepy.editor as mpy

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def infer_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Feedforward Inference function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup_feedforward()
    trainer.pipeline.eval()

    ## Initialize the volume and offset for each Gaussian Splatting point
    trainer.pipeline.model.init_volume()

    num_eval_images = len(trainer.pipeline.datamanager.eval_dataset)
    assert config.experiment_name is not None
    save_dir = "Zeroshot/"+config.experiment_name
    if Path(save_dir).is_dir():
            os.system(f"rm -rf {save_dir}")
    os.makedirs(save_dir,exist_ok=True)    

    lpips_fn = LearnedPerceptualImagePatchSimilarity().cuda()
    ssim = structural_similarity_index_measure

    sum_psnr = 0
    sum_lpips = 0
    sum_ssim = 0

    for i in tqdm(range(num_eval_images)):
        with torch.no_grad():
            outputs,gt_img = trainer.pipeline.get_eval_zeroshot(camera_idx = i)
        render_img = outputs['rgb']
        depth_map = colormaps.apply_XDLab_color_depth_map(outputs["depth"].squeeze().detach().cpu().numpy())
        gt_img =  torch.tensor(gt_img).to(outputs['accumulation'])
        media.write_image(os.path.join(save_dir,"render_{:03d}.png".format(i)), render_img.detach().cpu().numpy())
        media.write_image(os.path.join(save_dir,"depth_{:03d}.png".format(i)), depth_map)
        media.write_image(os.path.join(save_dir,"gt_{:03d}.png".format(i)), gt_img.squeeze().detach().cpu().numpy())
        
        psnr = -10. * np.log10(np.mean(np.square(gt_img.detach().cpu().numpy() - render_img.detach().cpu().numpy())))
        lpips = lpips_fn(gt_img.permute(0,3,1,2),render_img.unsqueeze(0).permute(0,3,1,2))
        SSIM = ssim(render_img.unsqueeze(0).permute(0,3,1,2),gt_img.permute(0,3,1,2))
        print("image {} PSNR:{:.4f} ,SSIM: {:.4f}, LPIPS: {:.4f}".format(i,psnr,SSIM,lpips))

        sum_psnr += psnr
        sum_lpips += lpips
        sum_ssim += SSIM   # type: ignore

    avg_psnr = sum_psnr/ num_eval_images
    avg_ssim = sum_ssim / num_eval_images
    avg_lpips = sum_lpips / num_eval_images

    CONSOLE.print(f"[bold green]Average PSNR:{avg_psnr}",justify="center")
    CONSOLE.print(f"[bold green]Average SSIM:{avg_ssim}",justify="center")
    CONSOLE.print(f"[bold green]Average Lpips:{avg_lpips}",justify="center")
    CONSOLE.print(f"[bold yellow]Save Dir in :{save_dir}")



def render_interpolate_camera(camera, trainer, save_dir:str, steps=100):
    """Render interpolated camera path and save video.

    Args:
        camera: Camera object containing camera parameters
        trainer: Model trainer instance
        save_dir: Directory to save rendered results
        steps: Number of interpolation steps between cameras
    """
    save_dir = "Zeroshot/inter_video/"+save_dir
    if Path(save_dir).is_dir():
        os.system(f"rm -rf {save_dir}")
    os.makedirs(save_dir,exist_ok=True)

    images_prob = []
    new_cameras = get_interpolated_camera_path(camera,steps=steps)
    trainer.pipeline.eval()
    for camera_idx in tqdm(range(new_cameras.size)):
        batch = trainer.pipeline.datamanager.get_interpolate_source_data(camera_idx, new_cameras)
        with torch.no_grad():
            outputs = trainer.pipeline.model.get_outputs_for_camera(new_cameras[camera_idx:camera_idx+1],batch)
        render_img = outputs['rgb']
        render_img = render_img[8:,...]
        images_prob.append(torch.cat([render_img],dim=0))

        ## Save rendered image as PNG file
        media.write_image(os.path.join(save_dir,"render_{:03d}.png".format(camera_idx)), render_img.detach().cpu().numpy())
    video = torch.stack(images_prob).permute(0,3,1,2).cpu()
    video = (video.clip(min=0, max=1) * 255).cpu().numpy()
    visualizations = {
            f"{save_dir}": wandb.Video(video[None], fps=15, format="mp4")
        }
    
    for key, value in visualizations.items():
        tensor = value._prepare_video(value.data) # type: ignore
        clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
        clip.write_videofile(
            f"{save_dir}.mp4", logger=None
        )


    exit()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and handles the
       training process on multiple processes.

    Args:
        local_rank: Current rank of process.
        main_func: Function that will be called by the distributed workers.
        world_size: Total number of gpus available.
        num_devices_per_machine: Number of GPUs per machine.
        machine_rank: Rank of this machine.
        dist_url: URL to connect to for distributed jobs, including protocol
            E.g., "tcp://127.0.0.1:8686".
            It can be set to "auto" to automatically select a free port on localhost.
        config: TrainerConfig specifying training regimen.
        timeout: Timeout of the distributed workers.

    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank

    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_devices_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comms.LOCAL_PROCESS_GROUP = pg

    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank)
    comms.synchronize()
    dist.destroy_process_group()
    return output


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one process
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(config: TrainerConfig) -> None:
    """Main function."""

    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.prompt:
        CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt

    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    config.set_timestamp()

    # print and save config
    # config.print_to_terminal()
    # config.save_config()

    launch(
        main_func=infer_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        ) # type: ignore
    )


if __name__ == "__main__":
    entrypoint()
