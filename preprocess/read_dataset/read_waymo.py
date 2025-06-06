import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import json
import cv2
from waymo_open_dataset import dataset_pb2 as open_dataset
from rich.console import Console

CONSOLE = Console(width=120)

class ReadWaymoData():
    def __init__(self, save_dir, sequence,root_dir):
        self.sequence = sequence
        self.save_root = save_dir
        self.data_root = root_dir
        self.inner_stop = False
        self.dir_name = None
        self.opengl2waymo = np.array([[0, 0, 1, 0],
                                    [-1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 0, 1]])
        type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
        ## Load all the scenes 
        self.num_seqs = os.listdir(self.data_root)
        CONSOLE.log(f"[red]Dataset: {len(self.num_seqs)} Waymo Sequence! \n")

    def generate_json(self,frame_start=80,num_frames=50):
        self.dir_name = os.path.join(self.save_root, 'seq_' + '{:02d}'.format(self.sequence) + '_waymo_' + str(frame_start).zfill(4) + f'_{num_frames}')
        os.makedirs(self.dir_name, exist_ok=True)
        CONSOLE.log(f"Create the BaseDir at {self.dir_name} ! \n")

        ## read the camera and image
        dataset = tf.data.TFRecordDataset(os.path.join(self.data_root,self.num_seqs[self.sequence]), compression_type='')
        cam2world, intrinsics,info = self.read_data(raw_dataset=dataset,frame_start=frame_start,num_frames=num_frames)
        return  cam2world, intrinsics, self.dir_name, info
    
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
    
    def read_data(self,raw_dataset,frame_start,num_frames):
        extr, intr, imsize = {}, {}, {}
        all_images = []
        all_images_name = []
        cams = [1] # no. of cameras
        for frame_idx, data in enumerate(tqdm(raw_dataset)):

            if (frame_idx < frame_start) or (frame_idx >= frame_start + num_frames):
                continue

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # images
            for img_pkg in frame.images:
                if img_pkg.name not in cams:
                    continue
                img = cv2.imdecode(np.frombuffer(img_pkg.image, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (960, 640), interpolation = cv2.INTER_CUBIC)
                all_images.append(img)
                all_images_name.append(frame_idx)
                h, w = img.shape[:2]
                if img_pkg.name not in imsize:
                    imsize[img_pkg.name] = []
                imsize[img_pkg.name].append((h, w))
            
            # ego pose
            v2w = np.array(frame.pose.transform).reshape(4,4)  # vehicle2global

            # camera calib
            for camera in frame.context.camera_calibrations:
                if camera.name not in cams:
                    continue
                if camera.name not in extr:
                    extr[camera.name] = []
                if camera.name not in intr:
                    intr[camera.name] = []

                cam_intrinsic = np.eye(4)
                cam_intrinsic[0, 0] = camera.intrinsic[0] / 2
                cam_intrinsic[1, 1] = camera.intrinsic[1] / 2
                cam_intrinsic[0, 2] = camera.intrinsic[2] / 2
                cam_intrinsic[1, 2] = camera.intrinsic[3] / 2
                intr[camera.name].append(cam_intrinsic)

                c2v = np.array(camera.extrinsic.transform).reshape(4, 4)  # cam2vehicle
                c2w = v2w @ c2v @ self.opengl2waymo
                extr[camera.name].append(c2w)

        
        imgs = np.stack(all_images, -1)
        imgs = np.moveaxis(imgs, -1, 0)
        poses = np.stack(extr[1])
        intrinsics = np.stack(intr[1])
        
        poses, inv_pose = self.pose_normalization(poses)
        
        def listify_matrix(matrix):
            matrix_list = []
            for row in matrix:
                matrix_list.append(list(row))
            return matrix_list
        
        H,W = imgs[0].shape[0], imgs[0].shape[1]
        out_data = {
            'fl_x': cam_intrinsic[0][0].item(),
            'fl_y': cam_intrinsic[1][1].item(),
            'cx': cam_intrinsic[0][2].item(),
            'cy': cam_intrinsic[1][2].item(),
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

        for i in range(poses.shape[0]):
            frame_data = {
                'file_path': f'./{str(all_images_name[i]).zfill(6)}.png',
                'transform_matrix': listify_matrix(poses[i]),
                'intrinsics': intrinsics[i].tolist(),
            }
            out_data['frames'].append(frame_data)
            filename = os.path.join(self.dir_name,f'{str(all_images_name[i]).zfill(6)}.png')
            cv2.imwrite(filename, all_images[i])

        with open(f'{self.dir_name}/transforms.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
            
        return poses,intrinsics,(H,W)

        

if __name__ == "__main__":
    reader = ReadWaymoData(save_dir='./waymo',sequence=0)
    reader.generate_json()
    exit()
