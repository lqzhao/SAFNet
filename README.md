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
 Currently, the code is not clean.
 The code is coming soon.
 
# Acknowledgements
We thank the authors of following works for opening source their excellent codes.
[MVPNet](https://github.com/maxjaritz/mvpnet)
[PointNet2](https://github.com/charlesq34/pointnet2)

# Citation
If you find our work useful, please cite our [paper](https://arxiv.org/abs/2107.01579):
```
@article{2107.01579,
Author = {Linqing Zhao and Jiwen Lu and Jie Zhou},
Title = {Similarity-Aware Fusion Network for 3D Semantic Segmentation},
Year = {2021},
journal={arXiv preprint arXiv:2107.01579},
}
```
