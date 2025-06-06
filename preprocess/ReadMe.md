### Prepare data
Download the [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) and [Waymo](https://waymo.com/open/) datasets, and place them in the appropriate root directories `ROOT_DIR`.

### Setup the environment
```bash
conda create --name Gendata -y python=3.8
conda activate Gendata
pip install --upgrade pip
```
#### Dependencies
Install PyTorch with CUDA.
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

#### Obtain the semantic and depth image
#### Metric depth
 We exploit the [Metric3d](https://github.com/YvanYin/Metric3D/tree/main) for metric depth predictions. 
Obtain the depth for each frame and put it into the `depth` file
#### Semantic mask
 We exploit the [hierarchical-multi-scale-attention](https://github.com/segcv/hierarchical-multi-scale-attention) for sky mask segmentation. 
 Obtain the semantic image for each frame and put it into the `semantics` file

**The dataset should have a structure as follows:**
```
├── $SCENE_0
    ├── depth
        ├──*.npy
    ├── semantic
        ├──instance/*.png
    ├── *.png
    ...
    ├── transfroms.json
```

### Running the Preprocessing Script
To preprocess specific scenes from the dataset, you can use the following commands. For example, to process 25 frames from `KITTI360` dataset starting at frame 8636 in sequence 4:

```shell
python run.py --seq_id=4 --start_index=8636 
--num_images=25 
--pcd_sparsity=Drop50 
--filter_sky 
--dataset=kitti360 
--save_dir=$SAVE_DIR$
--root_dir=$ROOT_DIR$
```

Alternatively, preprocess `waymo` dataset by providing the dataset type:
```shell
python run.py --seq_id=0 --start_index=110 
--num_images=50 
--pcd_sparsity=Drop50 
--filter_sky 
--dataset=waymo 
--save_dir=$SAVE_DIR$
--root_dir=$ROOT_DIR$
```

#### Parameter Description
```
| Parameter | Description |
|-----------|-------------|
| --seq_id | Sequence ID |
| --start_index | Starting frame index |
| --num_images | Number of images to process |
| --use_metric | Enable metric information |
| --use_semantic | Enable semantic information |
| --pcd_sparsity | Point cloud sparsity setting (Drop50 means 50% downsampling) |
| --filter_sky | Enable sky region filtering |
| --dataset | Dataset type (kitti360 or waymo) |
| --root_dir | Dataset root directory (required for Waymo dataset) |
| --save_dir | Output directory for processed results |
```