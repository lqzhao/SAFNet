[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/similarity-aware-fusion-network-for-3d/semantic-segmentation-on-scannet)](https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=similarity-aware-fusion-network-for-3d)[![arXiv](https://img.shields.io/badge/arXiv-2107.01579-b31b1b.svg)](https://arxiv.org/abs/2107.01579)

# SAFNet
Implementation of [Similarity-Aware Fusion Network for 3D Semantic Segmentation](https://arxiv.org/abs/2107.01579) IROS 2021

![](./pipeline.png)

## Environment Preparation & Data Preparation

We prepared our environment and [ScanNet data](http://kaldir.vc.in.tum.de/scannet_benchmark/) as follows: 

Environment: 

  - Python 3.6
  - Pytorch 1.2.0
  - CUDA 10.0 & CUDNN 7.6.4
 
DATA: 

  - The data is released under the [ScanNet Term of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf), please contact ScanNet team for access.
  - See [MVPNet](https://github.com/maxjaritz/mvpnet) repo for processing the raw data and resizing images.

<!-- ## Training -->

<!-- Pre-train 2D networks on the 2D semantic segmentation task.
```bash
python mvpnet/train_2d.py --cfg configs/scannet/unet_resnet34.yaml
```
 -->
 **Currently, the code is not clean.**
 
 The code is coming soon.
 
 ## Pre-trained Model
 
 We provide a pre-trained model (backbone: PointNet++ & ResNet34) which achieves **68.54% mIoU** and **88.07% Accuracy** on the validation set of ScanNetv2.
 
 The validation log was written in [this file](./log.test.07-07_22-52-27.ivg-221.txt).
 
 Please check the [BaiduDisk](https://pan.baidu.com/s/1-0TTaVea42OHyh8Z1tBBvw) with the code [f4n6].
 
 
## Core code

To see the corest part of our method, you can directly check [this file](./safnet/models/safnet_3d_late_fusion_attention_linear_mapping.py).
 
## Acknowledgements
We thank the authors of following works for opening source their excellent codes.

  - [MVPNet](https://github.com/maxjaritz/mvpnet)
  
  - [3DMV](https://github.com/angeladai/3DMV)

  - [PointNet2](https://github.com/charlesq34/pointnet2)

# Citation
If you find our work useful, please cite our [paper](https://arxiv.org/abs/2107.01579):
```
@inproceedings{zhao2021similarity,
  title={Similarity-Aware Fusion Network for 3D Semantic Segmentation},
  author={Zhao, Linqing and Lu, Jiwen and Zhou, Jie},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1585--1592},
  year={2021},
  organization={IEEE}
}
```
