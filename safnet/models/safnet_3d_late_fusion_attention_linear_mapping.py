import pdb
import torch
from torch import nn
import numpy as np
import math
import sys


from common.nn import SharedMLP, SharedMLPDO,MLP
from common.nn import LinearBNReLU
from safnet.models.pn2 import pn2ssg
from safnet.ops.group_points import group_points
from safnet.ops.ball_query import ball_query
from common.nn.functional import batch_index_select
from torch.nn.parameter import Parameter
from safnet.models.attention_method import eca_layer
from safnet.models.sift import PointSIFT_res_module,PointSIFT_module
from safnet.ops.knn_distance import knn_distance,knn_point

class FeatureAggregation(nn.Module):
    """Feature Aggregation inspired by ContFuse"""

    def __init__(self,
                 in_channels,
                 mlp_channels=(128, 64, 64),
                 reduction='sum',
                 use_relation=True,
                 ):
        super(FeatureAggregation, self).__init__()

        self.in_channels = in_channels
        self.use_relation = use_relation

        if mlp_channels:
            self.out_channels = mlp_channels[-1]
            self.mlp = SharedMLP(in_channels + (4 if use_relation else 0), mlp_channels, ndim=2, bn=True)
        else:
            self.out_channels = in_channels
            self.mlp = None

        if reduction == 'sum':
            self.reduction = torch.sum
        elif reduction == 'max':
            self.reduction = lambda x, dim: torch.max(x, dim)[0]

        self.reset_parameters()

    def forward(self, src_xyz, tgt_xyz, feature):
        """

        Args:
            src_xyz (torch.Tensor): (batch_size, 3, num_points, k)
            tgt_xyz (torch.Tensor): (batch_size, 3, num_points)
            feature (torch.Tensor): (batch_size, in_channels, num_points, k)

        Returns:
            torch.Tensor: (batch_size, out_channels, num_points)

        """
        if self.mlp is not None:
            if self.use_relation:
                k = src_xyz.shape[-1]
                d = 3
                diff_xyz = src_xyz[:,:,:,:d] - tgt_xyz.unsqueeze(-1)  # (b, 3, np, k)
                distance = torch.sum(diff_xyz ** 2, dim=1, keepdim=True)  # (b, 1, np, k)
                relation_feature = torch.cat([diff_xyz, distance],
                                             dim=1)
                x = torch.cat([feature[:,:,:,:d], relation_feature], 1)
            else:
                x = feature
            x = self.mlp(x)
            x = self.reduction(x, 3)
        else:
            x = self.reduction(feature, 3)
        return x

    def reset_parameters(self):
        from common.nn.init import xavier_uniform
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)

class DeepSimilarity(nn.Module):
    """real points                              xyz
        unprojected points                      unxyz
        unprojected feature(image feature)      unfeature"""
    def __init__(self,similarity_mode,radius=0.4,max_neighbors=16):
        super(DeepSimilarity, self).__init__()
        self.similarity_mode = similarity_mode
        self.maxradius = radius
        self.intradius = radius/2
        self.minradius = radius/4
        self.max_neighbors = max_neighbors
        self.int_neighbors = int(max_neighbors/2)
        self.min_neighbors = int(max_neighbors/4)
        if self.similarity_mode=='point':
            self.cos = torch.nn.CosineSimilarity(dim=1)

            self.pointsift_m1_1 = PointSIFT_module(radius=0.1, output_channel=128, extra_input_channel=192,
                                                        merge='add', same_dim=True)
            self.conv1d_2 = SharedMLP(128, (64,64 ),
                                       ndim=1, bn=True)

            self.fuse_weight_11 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_11.data.fill_(0.25) 
            self.fuse_weight_12 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_12.data.fill_(0.25)
            self.fuse_weight_21 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_21.data.fill_(0)   
            self.fuse_weight_22 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_22.data.fill_(0)   
            self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_3.data.fill_(1.0)   
            self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_4.data.fill_(0.1)   
            self.fuse_weight_5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_5.data.fill_(1.0)  
            self.fuse_weight_6 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_6.data.fill_(0.0)  

            self.linear1 = torch.nn.Linear(64, 64, bias=True)
            self.pointnet2 = pn2ssg.PN2SSGemb(64)
            self.linear2 = torch.nn.Linear(128, 128, bias=True)

    def forward(self, imageneighbors,knnpoints,points):
        b,_,n=points.shape
        imageneighbors = imageneighbors.permute(0,1,3,2)
        distance_1,distance_2 = self.geometric_similarity(knnpoints,imageneighbors,points)
        furthest_image_repre = imageneighbors.reshape(b, -1, n)
        try:
            furthest_real_repre = knnpoints.reshape(b,-1,n)
        except:
            furthest_real_repre = furthest_image_repre
        shallowimagefeat = self.pointsift_m1_1(points, furthest_image_repre)

        try:
            shallowrealfeat = self.pointsift_m1_1(points, furthest_real_repre)
        except:
            shallowrealfeat = shallowimagefeat

        shallowrealfeat = self.conv1d_2(shallowrealfeat)
        shallowimagefeat = self.conv1d_2(shallowimagefeat)

        shallowrealfeat1 = self.linear1(shallowrealfeat.permute(0, 2, 1)).permute(0, 2, 1)
        shallowimagefeat1 = self.linear1(shallowimagefeat.permute(0, 2, 1)).permute(0, 2, 1)

        shallow_similarity =  self.cos(shallowrealfeat1,shallowimagefeat1)
        #-------------------------------

        geometric_similarity_1 = torch.exp(-torch.reciprocal(self.fuse_weight_11)*distance_1)+self.fuse_weight_21
        geometric_similarity_2 = torch.exp(-torch.reciprocal(self.fuse_weight_12)*distance_2)+self.fuse_weight_22
        geometric_similarity = self.fuse_weight_5 * geometric_similarity_1 + self.fuse_weight_4 * geometric_similarity_2 + self.fuse_weight_6

        shallow_similarity = shallow_similarity * geometric_similarity

        # -------------------------------
        deeprealfeat = self.pointnet2({'points': points, 'feature': shallowrealfeat})

        if self.similarity_mode == 'point':

            deep_similarity = torch.ones_like(shallow_similarity) * self.fuse_weight_3

            return shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat

    def collect_neighbors(self, b,n,points,knnpoints,furthest_index,imageneighbors,furthest_index_img,max_neighbors):
        realneighbors = torch.ones(b,3,n,max_neighbors).cuda()
        imageneighbors2 = torch.ones(b, 3, n, max_neighbors).cuda()

        for j in range(b):
            realneighbors[j] = knnpoints[j,:,furthest_index[j,:,:]]
            imageneighbors2[j] = imageneighbors[j,:,furthest_index_img[j,:,:]]
        imageneighbors = imageneighbors2

        imageneifeat = self.neighbor_feature(imageneighbors,points)
        realneifeat = self.neighbor_feature(realneighbors,points)

        realneirepre = torch.cat([realneighbors, realneifeat], dim=1)     # b,7,n,k
        imageneirepre = torch.cat([imageneighbors, imageneifeat], dim=1)

        realneirepre = torch.reshape(realneirepre,(b,-1,n))
        imageneirepre = torch.reshape(imageneirepre,(b,-1,n))

        return realneirepre,imageneirepre,realneighbors,imageneighbors
    def neighbor_feature(self,neighbors, points):
        diff_xyz = neighbors - points.unsqueeze(-1)
        distance = torch.sum(diff_xyz ** 2, dim=1, keepdim=True)
        relation_feature = torch.cat([diff_xyz, distance], dim=1)
        return relation_feature
    def geometric_similarity(self,knn_realneighbors,knn_imageneighbors,points):
        a,b,c,d=knn_imageneighbors.shape
        knn_imageneighbors = knn_imageneighbors.permute(0,1,3,2)
        try:
            knn_realneighbors =knn_realneighbors.reshape(a,b,c,d)
            knn_realneighbors = knn_realneighbors.permute(0,1,3,2) #torch.Size([3, 3, 8192, 32])
        except:
            print(a,b,c,d)
            print(knn_realneighbors.shape)
            knn_realneighbors =knn_imageneighbors

        self.radius = 10

        real_diff = knn_realneighbors - points.unsqueeze(-1) # B,3,8192,k
        img_diff = knn_imageneighbors - points.unsqueeze(-1)
        real_distance = torch.sum(real_diff ** 2, dim=1, keepdim=True)
        img_distance = torch.sum(img_diff ** 2, dim=1, keepdim=True)

        real_mask = torch.ones_like(real_distance)
        real_mask[real_distance > self.radius**2] = 0
        real_mask_num = torch.sum(real_mask,dim=-1,keepdim=True)

        img_mask = torch.ones_like(img_distance)
        img_mask[img_distance > self.radius**2] = 0
        img_mask_num = torch.sum(img_mask,dim=-1,keepdim=True)

        dist1,_ = knn_point(1,knn_realneighbors,knn_imageneighbors)
        dist2,_ = knn_point(1,knn_imageneighbors,knn_realneighbors)

        dist1 = torch.abs(dist1)
        dist2 = torch.abs(dist2)

        avg_dist1 = torch.sum(dist1.squeeze(-1)*real_mask.squeeze(1),dim=-1) / real_mask_num.squeeze(1).squeeze(-1) + 1e-8
        avg_dist2 = torch.sum(dist2.squeeze(-1) * img_mask.squeeze(1), dim=-1) / img_mask_num.squeeze(1).squeeze(-1)
        avg_dist2[torch.isnan(avg_dist2)]=avg_dist1[torch.isnan(avg_dist2)]
        avg_dist2 = avg_dist2 + 1e-8

        return avg_dist1,avg_dist2
    def reset_parameters(self):
        from common.nn.init import xavier_uniform
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)

class MVPNet3D(nn.Module):
    def __init__(self, similarity,
                 net_2d,
                 net_2d_ckpt_path,
                 net_3d,
                 **feat_aggr_kwargs,
                 ):
        super(MVPNet3D, self).__init__()
        self.similarity_mode = similarity
        self.net_2d = net_2d
        # net_2d_ckpt_path=None
        if net_2d_ckpt_path:
            checkpoint = torch.load(net_2d_ckpt_path, map_location=torch.device("cpu"))
            a1,a2=self.net_2d.load_state_dict(checkpoint['model'], strict=False)
            # pdb.set_trace()
            import logging
            logger = logging.getLogger(__name__)
            logger.info("2D network load weights from {}.".format(net_2d_ckpt_path))
        self.feat_aggreg = FeatureAggregation(**feat_aggr_kwargs)

        self.similarity = DeepSimilarity(self.similarity_mode)
        # Attention Module
        self.atten_rgb = self.channel_attention(64)
        self.atten_point = self.channel_attention(128)

        self.narrow_conv1d = nn.Conv1d(192, 128, 1)
        self.mlp_seg = SharedMLPDO(128, (128,128,128), ndim=1, bn=True, p=0.5)
        self.seg_logit = nn.Conv1d(128, 20, 1, bias=True)


        self.mlp_seg_3d = SharedMLPDO(128, (128,128), ndim=1, bn=True, p=0.5)
        self.seg_logit_3d = nn.Conv1d(128, 20, 1, bias=True)


    def forward(self, data_batch):
        # (batch_size, num_views, 3, h, w)
        images = data_batch['images']
        depths = data_batch['depth']
        # labels_2d = data_batch['label_2d']
        b, nv, _, h, w = images.size()
        # collapse first 2 dimensions together

        images = images.reshape([-1] + list(images.shape[2:]))
        depths = depths.reshape([-1] + list(depths.shape[2:]))

        # 2D network
        preds_2d = self.net_2d({'image': images, 'depth': depths, 'label_2d':data_batch['label_2d']})
#         preds_2d = self.net_2d({'image': images, 'depth': depths, 'label_2d':depths})

        feature_2d = preds_2d['feature']  # (b * nv, c, h, w)

        logit_2d = preds_2d['seg_logit']

        # unproject features
        knn_indices = data_batch['knn_indices']  # (b, np, k) most cloes unprojected points
        feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
        feature_2d = feature_2d.reshape(b, -1, nv * h * w)
        feature_2d = group_points(feature_2d, knn_indices[:,:,:3])  # (b, c, np, k)
        
        # unproject depth maps
        with torch.no_grad():
            image_xyz = data_batch['image_xyz']  # (b, nv, h, w, 3)
            image_xyz = image_xyz.permute(0, 4, 1, 2, 3).reshape(b, 3, nv * h * w)
            image_xyz = group_points(image_xyz, knn_indices)  # (b, 3, np, k)

        # 2D-3D aggregation
        points = data_batch['points']
        knnpoints = data_batch['knnpoints']
        # down_sampled_knnpoints = data_batch['new_image_xyz_valid']
        b,_,n = points.shape

        feature_2d3d = self.feat_aggreg(image_xyz, points, feature_2d)

        shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat = self.similarity(image_xyz,knnpoints,points) # b,128,n

        stack_similarity = torch.stack([shallow_similarity,deep_similarity],0)
        stack_similarity = torch.nn.functional.softmax(stack_similarity, 0)
        shallow_similarity = stack_similarity[0,:]
        deep_similarity = stack_similarity[1,:]

        feature_2d3d = shallow_similarity.unsqueeze(1).expand(b, 64, n) * feature_2d3d
        weighted_deepfeature = deeprealfeat*deep_similarity.unsqueeze(1).expand(b, 128, n)
        atten_rgb = self.atten_rgb(feature_2d3d)
        atten_point = self.atten_point(weighted_deepfeature)
        
        feature_2d3d = feature_2d3d.mul(atten_rgb)
        weighted_deepfeature = weighted_deepfeature.mul(atten_point)

        fused_feature = torch.cat([feature_2d3d, weighted_deepfeature], 1)

        if self.similarity_mode == 'point':
            # attention

            fused_feature = torch.nn.functional.relu(self.narrow_conv1d(fused_feature))


            seg_feature = self.mlp_seg(fused_feature)
            seg_logit = self.seg_logit(seg_feature)

        seg_logit1 = logit_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, nc, nv, h, w)
        seg_logit1 = seg_logit1.reshape(b, -1, nv * h * w)
        seg_logit1 = group_points(seg_logit1, knn_indices[:,:,:1])  # (b, nc, np, k)
        seg_logit1 = seg_logit1.mean(-1)  # (b, nc, np)

        preds = dict()
        preds['logit_point_branch'] = self.seg_logit_3d(self.mlp_seg_3d(deeprealfeat))
        preds['logit_2d'] = logit_2d
        preds['logit_2d_chunks'] = seg_logit1
        preds['seg_logit'] = seg_logit

        return preds,None,preds_2d['depth_loss']



    def channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool1d(1)
        conv = nn.Conv1d(num_channel, num_channel, kernel_size=1)
        # bn = nn.BatchNorm2d(num_channel)
        activation = nn.Sigmoid() # todo modify the activation function

        return nn.Sequential(*[pool, conv, activation])

    def unproject(self, feature_2d, scale_factor,b,nv,h, w):
        newh = int(h/scale_factor)
        neww = int(w/scale_factor)
        # pdb.set_trace()
        feature_2d = feature_2d.reshape(b, nv, -1, newh, neww).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
        feature_2d = feature_2d.reshape(b, -1, nv * newh * neww)
        # feature_2d = group_points(feature_2d, knn_indices)  # (b, c, np, k)
        return feature_2d

    def get_loss(self, cfg):
        from safnet.models.loss import SegLoss, OhemCELoss
        if cfg.TRAIN.LABEL_WEIGHTS_PATH:
            weights = np.loadtxt(cfg.TRAIN.LABEL_WEIGHTS_PATH, dtype=np.float32)
            weights = torch.from_numpy(weights).cuda()
        else:
            weights = None
        return SegLoss(weight=weights)
        # return OhemCELoss(0.5, n_min=8192, weight=weights,ignore_index=-100)

    def get_metric(self, cfg):
        from safnet.models.metric import SegAccuracy, SegIoU
        metric_fn = lambda: [SegAccuracy(), SegIoU(20)]
        return metric_fn(), metric_fn()

def show_cam_on_image(img, mask):
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 0.5*heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("/home/zlq/example.jpg", np.uint8(255 * cam))

def plot_geometry_distance(pc,distance):
    # pc = pc.permute(2,1,3,0).squeeze().cpu()
    # distance = distance.squeeze().cpu()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    pc = pc.permute(2,1,0).squeeze().cpu()
    distance = distance.permute(1,0).cpu()

    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], c=distance[:,0],s=1)
    # ax.axis('off')

    cbar = plt.colorbar(p)

    plt.show()
    pdb.set_trace()

def plot_geometry_distance_subplot(pc,distance1,distance2,distance):
    # pc = pc.permute(2,1,3,0).squeeze().cpu()
    # distance = distance.squeeze().cpu()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    pc = pc.permute(2,1,0).squeeze().cpu()
    distance1 = distance1.permute(1,0).cpu()
    distance2 = distance2.permute(1, 0).cpu()
    distance = distance.permute(1, 0).cpu()
    pdb.set_trace()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax = fig.add_subplot(1,2,1,projection='3d')
    p = ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], c=distance[:,0],s=1)
    # ax.axis('off')

    cbar = plt.colorbar(p)

    plt.show()
    pdb.set_trace()
