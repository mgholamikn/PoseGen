# PoseGen: Learning to Generate 3D Human Pose Dataset with NeRF
[[AAAI 2024] PoseGen: Learning to Generate 3D Human Pose Datasets with NeRF](https://arxiv.org/pdf/2312.14915.pdf)

![Funny Cat](Figures/PoseGenFramework.jpg)




### Setup environment

```
conda create -n posegen python=3.8
conda activate posegen

# install pytorch for your corresponding CUDA environments
pip install torch

# install pytorch3d: note that doing `pip install pytorch3d` directly may install an older version with bugs. be sure that you specify the version that matches your CUDA environment. See: https://github.com/facebookresearch/pytorch3d
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt190/download.html

# install other dependencies
pip install -r requirements.txt

```
### Download pre-processed data and pre-trained models
Please download the data and log zip files from: https://drive.google.com/drive/folders/1Pxvo1o_tKkptNVB-vAs-RnyCSwDpKJyP?usp=sharing 

unzip logs.zip and data.zip in the main directory. There should be a data/ and logs/ in the main directory.

### Training 
To train PoseGen to generate data with SPIN as the baseline model:

```
python run_gan.py --nerf_args configs/surreal/surreal.txt --ckptpath logs/surreal_model/surreal.tar  --dataset surreal --entry hard  --runname render_3dpw_testset --white_bkgd  --render_res 512 512

```

Citation:
```
@misc{gholami2023posegen,
      title={PoseGen: Learning to Generate 3D Human Pose Dataset with NeRF}, 
      author={Mohsen Gholami and Rabab Ward and Z. Jane Wang},
      year={2023},
      eprint={2312.14915},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
