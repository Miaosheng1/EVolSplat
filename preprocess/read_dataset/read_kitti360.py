import numpy as np
import os
import imageio.v2 as imageio
import json
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from itertools import chain
from rich.console import Console
from copy import deepcopy

CONSOLE = Console(width=120)

class ReadKITTI360Data():
    def __init__(self, save_dir, sequence,spars,root_dir):
        self.sequence = sequence
        self.save_root = save_dir
        self.data_root = root_dir
        self.inner_stop = False
        self.dir_name = None
        self.spars = spars

    def generate_json(self,frame_start,num_frames=40):
        seq_id = self.sequence[-7:-5]
        self.dir_name = os.path.join(self.save_root, 'seq_' + seq_id + '_nerfacto_' + str(frame_start).zfill(4) + f'_{num_frames}')
        os.makedirs(self.dir_name, exist_ok=True)
        CONSOLE.log(f"Create the BaseDir at {self.dir_name} ! \n")

        ## read the camera and image
        cam2world_dict_00, cam2world_dict_01, info = self.read_data(frame_start,num_frames)
        return cam2world_dict_00, cam2world_dict_01, self.dir_name, info
    
    def pose_normalization(self, poses):
        cameara_type_matrix = np.array([1, -1, -1])

        mid_frames = poses.shape[0] // 2 
        inv_pose = np.linalg.inv(poses[mid_frames])
        for i, pose in enumerate(poses):
            if i == mid_frames:
                poses[i] = np.eye(4)
            else:
                poses[i] = np.dot(inv_pose, poses[i])  # Note: inv_pose is left-multiplied
        
        for i in range(poses.shape[0]):
            poses[i, :3, :3] = poses[i, :3, :3] * cameara_type_matrix  ## opencv2openGL
    
        return poses, inv_pose
    
    def loadCameraToPose(self,filename):
        # open file
        Tr = {}
        lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                lineData = list(line.strip().split())
                if lineData[0] == 'image_01:':
                    data = np.array(lineData[1:]).reshape(3,4).astype(np.float32)
                    data = np.concatenate((data,lastrow), axis=0)
                    Tr[lineData[0]] = data
                elif lineData[0] == 'image_02:':
                        data = np.array(lineData[1:]).reshape(3, 4).astype(np.float32)
                        data = np.concatenate((data, lastrow), axis=0)
                        Tr[lineData[0]] = data
                elif lineData[0] == 'image_03:':
                    data = np.array(lineData[1:]).reshape(3, 4).astype(np.float32)
                    data = np.concatenate((data, lastrow), axis=0)
                    Tr[lineData[0]] = data
        return Tr
    

    def read_data(self,frame_start,num_frames):
        intrinstic_file = os.path.join(os.path.join(self.data_root, 'calibration'), 'perspective.txt')
        with open(intrinstic_file) as f:
            lines = f.readlines()
            for line in lines:
                lineData = line.strip().split()
                if lineData[0] == 'P_rect_00:':
                    K_00 = np.array(lineData[1:]).reshape(3,4).astype(np.float32)
                    K_00 = K_00[:,:-1]
                elif lineData[0] == 'P_rect_01:':
                    K_01 = np.array(lineData[1:]).reshape(3,4).astype(np.float32)
                    K_01 = K_01[:,:-1]
                elif lineData[0] == 'R_rect_01:':
                    R_rect_01 = np.eye(4)
                    R_rect_01[:3,:3] = np.array(lineData[1:]).reshape(3,3).astype(np.float32)

        CamPose_00 = {}
        CamPose_01 = {}
        
        extrinstic_file = os.path.join(self.data_root, os.path.join('data_poses', self.sequence))
        cam2world_file_00 = os.path.join(extrinstic_file, 'cam0_to_world.txt')
        pose_file = os.path.join(extrinstic_file, 'poses.txt')

        ''' Camera_00 to world coordinate '''
        with open(cam2world_file_00,'r') as f:
            lines = f.readlines()
            for line in lines:
                lineData = list(map(float,line.strip().split()))
                CamPose_00[lineData[0]] = np.array(lineData[1:]).reshape(4,4)

        ''' Camera_01 to world coordiante '''
        CamToPose = self.loadCameraToPose(os.path.join(os.path.join(self.data_root, 'calibration'), 'calib_cam_to_pose.txt'))
        CamToPose_01 = CamToPose['image_01:']
        poses = np.loadtxt(pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
            pp = np.matmul(pose, CamToPose_01)
            CamPose_01[frame] = np.matmul(pp, np.linalg.inv(R_rect_01))
        
        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)
        
        imgae_dir = os.path.join(self.data_root, self.sequence)
        image_00 = os.path.join(imgae_dir, 'image_00/data_rect')
        image_01 = os.path.join(imgae_dir, 'image_01/data_rect')

        all_images = []
        all_poses = []
        all_img_name = []

        for idx in tqdm(range(frame_start, frame_start+num_frames, 1)):
            
            if idx not in CamPose_00.keys() or idx not in CamPose_01.keys():
                self.inner_stop = True
                return None, None
            
            ## read image_00
            image = imread(os.path.join(image_00, "{:010d}.png").format(idx))
            all_images.append(image)
            all_poses.append(CamPose_00[idx])
            all_img_name.append(str(idx) + "_00")

            ## read image_01
            image = imread(os.path.join(image_01, "{:010d}.png").format(idx))
            all_images.append(image)
            all_poses.append(CamPose_01[idx])
            all_img_name.append(str(idx) + "_01")
        
        imgs = np.stack(all_images, -1)
        imgs = np.moveaxis(imgs, -1, 0)
        poses = np.stack(all_poses)
        
        poses, inv_pose = self.pose_normalization(poses)
        
        def listify_matrix(matrix):
            matrix_list = []
            for row in matrix:
                matrix_list.append(list(row))
            return matrix_list
        
        poses = poses[:imgs.shape[0]]
        H,W = imgs[0].shape[0], imgs[0].shape[1]
        out_data = {
            'fl_x': K_00[0][0].item(),
            'fl_y': K_00[0][0].item(),
            'cx': K_00[0][2].item(),
            'cy': K_00[1][2].item(),
            'w': imgs[0].shape[1],
            'h': imgs[0].shape[0],
            "inv_pose": listify_matrix(inv_pose),
            "scale": 1,
            'ply_file_path_Drop25': f"Drop25/{frame_start}.ply",
            'ply_file_path_Drop50': f"Drop50/{frame_start}.ply",
            'ply_file_path_full': f"full/{frame_start}.ply",
            'ply_file_path_Drop80': f"Drop80/{frame_start}.ply",
            }
        out_data['frames'] = []

        for i in range(0, poses.shape[0]):
            frame_data = {
                'file_path': all_img_name[i] + ".png",
                'transform_matrix': listify_matrix(poses[i])
            }
            filename =f'{self.dir_name}/' + all_img_name[i] + ".png"
            imageio.imwrite(filename, imgs[i])
            out_data['frames'].append(frame_data)

        with open(f'{self.dir_name}/transforms.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
            
        return CamPose_00, CamPose_01, (K_00,H,W)

        

