import numpy as np
import os
from argparse import ArgumentParser
from read_dataset.read_kitti360 import ReadKITTI360Data
from read_dataset.read_waymo import ReadWaymoData
from typing import Literal, Optional
from PIL import Image
import cv2
from rich.console import Console
from read_dataset.generate_pcd import PCDGenerator
from read_dataset.generate_waymo_pcd import WaymoPCDGenerator

CONSOLE = Console(width=120)


class ParamGroup:
    def __init__(self):
        return

    def extract(self, args):
        for arg in vars(args).items():
            setattr(self, arg[0], arg[1])


class System():
    pcd_sparsity: Optional[Literal['Drop90','Drop50',"Drop80","Drop25","full"]] = None
    """ the pointcloud generation must be in 5 types: Drop90, Drop80, Drop50, full, none"""

    depth_cosistency: bool =  True

    def __init__(self,save_dir,sys_augments) -> None:
        self.seq_id = '2013_05_28_drive_{:04d}_sync'.format(sys_augments.seq_id)
        
        self.frame_start = sys_augments.start_index
        self.use_semantics = sys_augments.use_semantic

        self.use_metric = sys_augments.use_metric_depth
        self.filter_pcd_sky = sys_augments.filter_sky
        self.pcd_sparsity = sys_augments.pcd_sparsity

        self.dir_name = None
        # Point cloud downsampling configuration
        # Full image: downsampling rate of 3, approximately 2M GS points per scene
        # Drop50, Drop25: downsampling rate can be set to 2, approximately 2M points per scene
        self.downscale_pcd = 1   

        if sys_augments.dataset == 'waymo':
            self.data_reader = ReadWaymoData(save_dir=save_dir,sequence= sys_augments.seq_id,root_dir=sys_augments.root_dir)
            self.pcd_generator = WaymoPCDGenerator(spars=self.pcd_sparsity,
                                            save_dir= sys_augments.pcd_sparsity,
                                            frame_start=self.frame_start,
                                            filer_sky= self.filter_pcd_sky,
                                            depth_cosistency=True,
                                            )
        elif sys_augments.dataset == 'kitti360':
            self.data_reader = ReadKITTI360Data(save_dir=save_dir,sequence=self.seq_id, spars=self.pcd_sparsity,root_dir=sys_augments.root_dir)
   
            self.pcd_generator = PCDGenerator(spars=self.pcd_sparsity,
                                            save_dir= sys_augments.pcd_sparsity,
                                            frame_start=self.frame_start,
                                            filer_sky= self.filter_pcd_sky,
                                            depth_cosistency=self.depth_cosistency,
                                            )
        CONSOLE.print(f"Sparity: {sys_augments.pcd_sparsity} !",justify="center")

        
      
    def forward(self,num_frames):
        """This function is designed for KITTI-360"""
        cam2world_dict_00, cam2world_dict_01, self.dir_name, info = self.data_reader.generate_json(frame_start=self.frame_start,num_frames=num_frames)
        self.pcd_generator.set_image_info(info=info)
        self.H = info[1]
        self.W = info[2]

        # Use stereo or monocular depth
        if self.use_metric and not os.path.exists(os.path.join(self.dir_name,"depth")):
            self.gen_metric_depth()
        else:
            CONSOLE.log(" Skip depth output")

        # whether output semantic map
        if self.use_semantics and not os.path.exists(os.path.join(self.dir_name,"semantic")):
            self.gen_semantic()
            self.gen_sky_mask()
  
        # whether accumulate the pointcloud
        if self.filter_pcd_sky:
            assert (self.use_semantics == True) or (os.path.exists(os.path.join(self.dir_name,"semantic")))

        if self.pcd_sparsity is None:
            return
        else:
            self.pcd_generator.forward(dir_name=self.dir_name,
                                    cam2world_dict_00=cam2world_dict_00,
                                    cam2world_dict_01=cam2world_dict_01,
                                    down_scale=self.downscale_pcd)

    def Waymo_forward(self,num_frames):
        """This function is designed for Waymo"""        
        poses, intrinsics, self.dir_name, info = self.data_reader.generate_json(frame_start=self.frame_start,num_frames=num_frames)
        self.H = info[0]
        self.W = info[1]

        if self.use_metric and not os.path.exists(os.path.join(self.dir_name,"depth")):
            self.gen_metric_depth(dataset='waymo')
        else:
            CONSOLE.log(" Skip depth output")

        # whether output semantic map
        if self.use_semantics and not os.path.exists(os.path.join(self.dir_name,"semantic")):
            self.gen_semantic()
            self.gen_sky_mask()
        else:
            CONSOLE.log(" Skip semantic output")
  
        # whether accumulate the pointcloud
        if self.filter_pcd_sky:
            assert (self.use_semantics == True) or (os.path.exists(os.path.join(self.dir_name,"semantic")))

        if self.pcd_sparsity is None:
            return
        else:
            self.pcd_generator.forward(dir_name=self.dir_name,
                                    poses=poses,
                                    intrinsics=intrinsics,
                                    H=self.H,
                                    W=self.W,
                                    down_scale=self.downscale_pcd,
                                    )
        

    def gen_metric_depth(self,dataset='kitti360'):
        """We use the metric DepthV2 Giant Model. When predict the waymo, we need to run on the other GPU"""
        # Note: dataset is specified as kitti360 because metric3d code uses dataset type to determine intrinsics
        # If you need to modify intrinsics, please modify the original code in mono/utils/custom_data.py
        metric3d_path = os.getenv('METRIC3D_PATH', '/home/smiao/Gen_Dataset/dataset_methods/metric3d')
        model_path = os.getenv('METRIC3D_MODEL_PATH', '/nas/users/smiao/model_zoo/metric3d/metric_depth_vit_giant2_800k.pth')
        gpu_id = os.getenv('DEPTH_GPU_ID', '6')
        
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {metric3d_path}/mono/tools/test_scale_cano.py \
            {metric3d_path}/mono/configs/HourglassDecoder/vit.raft5.giant2.py \
            --load-from {model_path} \
            --test_data_path {self.dir_name} --show-dir {os.path.join(self.dir_name, 'depth')}\
            --dataset {dataset}\
            --launcher None"    
        os.system(cmd)

    def gen_semantic(self):
        nvi_sem_path = os.getenv('NVI_SEM_PATH', '/home/smiao/Gen_Dataset/dataset_methods/nvi_sem')
        checkpoint_path = os.getenv('NVI_SEM_CHECKPOINT', '/home/smiao/Gen_Dataset/dataset_methods/nvi_sem/checkpoints/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth')
        gpu_id = os.getenv('SEMANTIC_GPU_ID', '6')
        
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {nvi_sem_path}/train.py --dataset cityscapes --cv 0 \
               --bs_val 1 --eval folder --n_scales 1.0 --dump_assets --dump_all_images \
               --snapshot {checkpoint_path}\
               --arch ocrnet.HRNet_Mscale --result_dir {os.path.join(self.dir_name, 'semantic')} --eval_folder {self.dir_name}"
        
        os.system(cmd)
        # Clean up unnecessary files, keep only folders
        os.system(f"find {os.path.join(self.dir_name, 'semantic')} -maxdepth 1 -type f -delete")

    def gen_sky_mask(self):
        instance_path = os.path.join(self.dir_name, 'semantic', 'instance')
        save_path = os.path.join(self.dir_name, 'mask')
        os.makedirs(save_path,exist_ok=True)
        image_height, image_width = self.H, self.W
     
        for instance_fn in sorted(os.listdir(instance_path)):
            mask = np.ones((image_height, image_width), dtype=np.uint8)
            instance_file = os.path.join(instance_path, instance_fn)
            instance = cv2.imread(instance_file, -1)
            mask[instance==10] = 0
            mask = Image.fromarray(mask * 255)
            mask.save(os.path.join(save_path, instance_fn))

if __name__ == "__main__":
    parser = ArgumentParser(description="Datasets parameters")
    parser.add_argument('--start_index',type = int,help=' Start index of Data',default=0)
    parser.add_argument('--num_images',type = int,help='Num images',default=40)
    parser.add_argument('--seq_id',type = int,help='kitti360 sequence_id',default=0)
    parser.add_argument('--use_semantic',action='store_true')
    parser.add_argument('--use_metric_depth',action='store_true')
    parser.add_argument('--filter_sky',action='store_true')
    parser.add_argument('--root_dir',type=str,default='/data0/datasets/KITTI-360/')
    parser.add_argument('--save_dir',type=str,default='/data1/smiao/waymo_new_infer')
    parser.add_argument('--pcd_sparsity', type=str, choices=['Drop90', 'Drop50', 'Drop80', 'Drop25','full'], help='Point cloud sparsity')
    parser.add_argument('--dataset',type=str,choices=['kitti360','waymo'],default='kitti360')

    config = parser.parse_args()
    save_idr = config.save_dir
    assert config.root_dir is not None, "Please specify the root directory of the dataset"

    sys_augments = ParamGroup()
    sys_augments.extract(config)
    system = System(save_dir=save_idr,sys_augments=sys_augments)

    if config.dataset == 'waymo':
         system.Waymo_forward(num_frames=config.num_images)
    elif config.dataset == 'kitti360':
        system.forward(num_frames=config.num_images)
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")