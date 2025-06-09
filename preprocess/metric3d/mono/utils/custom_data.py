import glob
import os
import json
from pathlib import Path
import numpy as np

def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb']
        depth = anno['depth'] if 'depth' in anno else None
        depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
        intrinsic = anno['cam_in'] if 'cam_in' in anno else None

        data_i = {
            'rgb': rgb,
            'depth': depth,
            'depth_scale': depth_scale,
            'intrinsic': intrinsic,
            'filename': os.path.basename(rgb),
            'folder': rgb.split('/')[-3],
        }
        datas.append(data_i)
    return datas

def load_from_json(filename: Path):
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)
    
def load_intrinsics(path:str):
    " Return intrinsics list [fl_x, fl_x, cx, cy] "
    meta = load_from_json(path)
    intris = []
    # for sub_i,frame in enumerate(meta["frames"]):
    #     K = frame['intrinsics']
        # intris.append(np.stack([K[0][0],K[1][1],K[0][2],K[1][2]]))
    return np.stack([meta['fl_x'],meta['fl_y'],meta['cx'], meta['cy']])

def load_data(path: str, dataset: str):
    rgbs = sorted(glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png') + 
                  glob.glob(path + '/front_images'+ '/*.jpg') + glob.glob(path + '/front_images'+ '/*.png'))
    # print("This is the rgb path: ", path)
    # exit()
    if dataset == 'kitti360':
        intrinsics = [552.5542602539062, 552.5542602539062, 682.0494384765625, 238.76954650878906]
    else:
        assert os.path.exists(os.path.join(path,"transforms.json"))
        intrinsics = load_intrinsics(os.path.join(path,"transforms.json"))
        # intrinsic = [1038.8481920789602, 1038.8481920789602, 471.19850280484064, 324.1388703680434]

    data = [{'rgb':rgb, 'depth':None,  'intrinsic': intrinsics, 'filename':os.path.basename(rgb), 
             'folder': rgb.split('/')[-3]} for i, rgb in enumerate(rgbs)]
    return data