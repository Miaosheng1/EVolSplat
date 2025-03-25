<h1 align="center">EVolSplat: Efficient Volumetric Splatting for Real-Time Urban View Synthesis (CVPR 2025)</h1>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2407.12395)

Sheng Miao, [Jiaxin Huang](https://jaceyhuang.github.io/), Dongfeng Bai, Xu Yan, [Hongyu Zhou](https://hyzhou404.github.io/), [Yue Wang](https://ywang-zju.github.io/), Bingbing Liu, [Andreas Geiger](https://www.cvlibs.net/) and [Yiyi Liao](https://yiyiliao.github.io/) 

Our project page can be seen [here](https://xdimlab.github.io/EVolSplat/).
<img src="./docs/teaser.png" height="200">
## :book: Datasets
We evaluate our model on [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/) and [Waymo](https://waymo.com/open/download/). Here we show the structure of a test dataset as follow, similar to the [EDUS](https://xdimlab.github.io/EDUS/). 
We provide the example data for inference on KITTI-360, which can be found in huggingface [here](https://huggingface.co/datasets/cookiemiao/EVolSplat_infer_dataset/tree/main).


The dataset should have a structure as follows:
```
├── $PATH_TO_YOUR_DATASET
    ├── $SCENE_0
        ├── depth
        ├── pointcloud/*.ply
        ├── *.png
        ...
        ├── transfroms.json
    ...
    ├── SCENE_N
        ├── depth
        ├── pointcloud/*.ply
        ├── *.png
        ...
        ├── transfroms.json
```

## :house: Installation
Our EVolSplat is built on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). You can follow the nerfstudio webpage to install our code.  


#### Create environment
We recommend using conda to create a new environment and you can find the detailed environment file in `environment.yml`.
```bash
conda create --name EVolSplat -y python=3.8
conda activate EVolSplat
pip install --upgrade pip
```
#### Dependencies
Install PyTorch with CUDA (this repo has been tested with CUDA 11.8).
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
After pytorch, install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):
```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

```
Install the spaseCNN library, we recommend the version 2.1.0. We can find the installation instruction in the [torchsparse](https://github.com/mit-han-lab/torchsparse) repository.
```bash
conda install -c conda-forge sparsehash
sudo apt-get install libsparsehash-dev  
git clone --recursive https://github.com/mit-han-lab/torchsparse
python setup.py install
```

#### Installing EVolSplat
Install EVolSplat form source code
```bash
git clone https://github.com/XDimLab/EVolSplat.git
cd EVolSplat
pip install --upgrade pip setuptools
pip install -e .
```




## :chart_with_upwards_trend: Evaluation & Checkpoint
We provide the pretrained model trained on `KITTI-360` and `Waymo` and you can download the pre-trained models from  [here](https://xdimlab.github.io/EVolSplat/). 

Place the downloaded checkpoints in `checkpoint` folder in order to test it later.

### Feed-forward Inference
Replace `$PATH_TO_YOUR_DATASET$` with your data path.
```
python nerfstudio/scripts/infer_zeroshot.py evolsplat \
  --load_dir checkpoints/ \
  --pipeline.model.freeze_volume=True \
  zeronpt-data \
  --data $PATH_TO_YOUR_DATASET$ \
  --kitti=True 
```


## :clipboard: Citation

If our work is useful for your research, please give me a star and consider citing:

```
@inproceedings{miao2025efficient,
  title={EVolSplat: Efficient Volumetric Splatting for Real-Time Urban View Synthesis},
  author={Miao, Sheng and Huang, Jiaxin and Bai, Dongfeng and Yan, Xu and Zhou, Hongyu and Wang, Yue and Liu, Bingbing and Geiger, Andreas and Liao, Yiyi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
}
```
## :sparkles: Acknowledgement
- This project is based on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- Some codes are brought from [IBRNet](https://github.com/googleinterns/IBRNet).