
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import cv2
from nerfstudio.utils import colormaps
from PIL import Image
from pathlib import Path

TINY_NUMBER = 1e-6

def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1,
                        keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1,
                        keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(
        np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists

def read_rgb_filename(image_filename=None):
    assert image_filename is not None
    pil_image = Image.open(image_filename)
    image = np.array(pil_image, dtype="uint8")
    image = torch.from_numpy(image.astype("float32") / 255.0)
    return image

def get_image_depth_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = np.load(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.shape
        new_width, new_height = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = cv2.resize(pil_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    depth_tensor = torch.from_numpy(np.array(pil_mask))
    return depth_tensor

def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))

def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0),
                         view_selection_method='nearest',
                         view_selection_stride=None,
                         ):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    # num_select = 4
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(
            batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)

    if view_selection_method == 'nearest':
        if view_selection_stride is not None:
            idx = np.minimum(np.arange(1, num_select + 1, dtype=int)
                             * view_selection_stride, num_cams - 1)
            selected_ids = sorted_ids[idx]
        else:
            selected_ids = sorted_ids[:num_select]
    else:
        raise Exception('unknown view selection method!')

    return selected_ids

def find_corres_index(image_idx,indices):
    res = []
    for value in indices:
        index = torch.where(image_idx == value)[0]
        res.append(index.detach().cpu())
    res = torch.cat(res)
    return res


def get_source_images_from_current_imageid(image_id,
                                           all_pose, 
                                           rgbs,
                                           depths = None,
                                           num_select = 2):

    eye = torch.tensor([0., 0., 0., 1.]).to(all_pose)
    all_pose = torch.cat([all_pose,eye[None,None,:].repeat(all_pose.shape[0],1,1)],dim=1)
    target_pose = all_pose[image_id]

    nearest_pose_ids = get_nearest_pose_ids(target_pose.detach().cpu().numpy(),
                         all_pose.detach().cpu().numpy(),
                         num_select=num_select,
                         tar_id=image_id,
                         angular_dist_method='dist',
                         )
    ## sorted the nearest id
    nearest_pose_ids = np.array(sorted(nearest_pose_ids))
    src_poses = all_pose[nearest_pose_ids,...]
    src_rgbs = [read_rgb_filename(rgbs[i]) for i in nearest_pose_ids]
    src_rgbs =  torch.stack(src_rgbs,dim=0)

    if depths is not None:
        src_depths = [get_image_depth_tensor_from_path(depths[i]) for i in nearest_pose_ids]
        src_depths =  torch.stack(src_depths,dim=0)
    else:
        src_depths = None

    return src_rgbs, src_poses, target_pose, nearest_pose_ids, src_depths


def eval_source_images_from_current_imageid(rgbs,
                                            depths,
                                           all_pose, 
                                           eval_pose,
                                           num_select = 2):

    eye = torch.tensor([0., 0., 0., 1.]).to(all_pose)
    all_pose = torch.cat([all_pose,eye[None,None,:].repeat(all_pose.shape[0],1,1)],dim=1)
    eval_pose = torch.cat([eval_pose,eye[None,None,:].repeat(eval_pose.shape[0],1,1)],dim=1)
    assert len(eval_pose) == 1

    ## nearest_ids should be in training data_cache
    ## tar_id= -1, 因为我们在 train_cameras 找，可以包含所有的 input camera
    nearest_pose_ids = get_nearest_pose_ids(eval_pose.detach().cpu().numpy()[0],
                         all_pose.detach().cpu().numpy(),
                         num_select=num_select,
                         tar_id= -1,
                         angular_dist_method='dist',
                         )
    nearest_pose_ids = np.array(sorted(nearest_pose_ids))
    src_poses = all_pose[nearest_pose_ids,...]
    src_rgbs = [read_rgb_filename(rgbs[i]) for i in nearest_pose_ids]
    src_rgbs =  torch.stack(src_rgbs,dim=0)

    if depths is not None:
        src_depths = [get_image_depth_tensor_from_path(depths[i]) for i in nearest_pose_ids]
        src_depths =  torch.stack(src_depths,dim=0)
    else:
        src_depths = None
    

    return src_rgbs, src_poses, nearest_pose_ids, src_depths


def render_trajectory_source_pose(image_batch,
                                    scene_id,
                                    all_pose, 
                                    target_pose,
                                    num_select = 3,
                                    num_imgs_per_scene=0):
    assert num_imgs_per_scene > 0
    """(N,4,4) pose matrices"""
    eye = torch.tensor([0., 0., 0., 1.]).to(all_pose)
    all_pose = torch.cat([all_pose,eye[None,None,:].repeat(all_pose.shape[0],1,1)],dim=1)
    target_pose = torch.cat([target_pose,eye[None,:]],dim=0)
   
    start_pose_id = num_imgs_per_scene * scene_id
    end_pose_id = num_imgs_per_scene* (scene_id+1)
    train_poses = all_pose[start_pose_id:end_pose_id,:,:]
    
    nearest_pose_ids = get_nearest_pose_ids(target_pose.detach().cpu().numpy(),
                         train_poses.detach().cpu().numpy(),
                         num_select=num_select,
                         tar_id=-1,
                         angular_dist_method='dist',
                         )
    nearest_pose_ids = np.array(sorted(nearest_pose_ids))
    nearest_pose_ids = nearest_pose_ids + scene_id*num_imgs_per_scene
    src_poses = all_pose[nearest_pose_ids,...]
    nearest_pose_ids = find_corres_index(image_batch['image_idx'],nearest_pose_ids)
    src_rgbs = image_batch['image'][nearest_pose_ids]  ## fix index 之后才可以检索图像

    if 'depth' in image_batch:
        src_depths = image_batch['depth'][nearest_pose_ids]  ## fix index 之后才可以检索图像
    else:
        src_depths = None
    src_poses = torch.cat([src_poses,target_pose.unsqueeze(dim=0)])
    return src_rgbs,src_poses,src_depths


