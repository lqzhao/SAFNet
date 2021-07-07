import pdb
import torch
from torch import nn
import numpy as np
import math
import sys


# sys.path.append('/home/syy/zlq/GACNet')
# from model import GACNet

# sys.path.append('/home/syy/zlq/Pointnet_Pointnet2_pytorch/models')
# from pointnet2_sem_seg_msg import get_model

from common.nn import SharedMLP, SharedMLPDO
# from common.nn import LinearBNReLU
from safnet.models.pn2 import pn2ssg
from safnet.ops.group_points import group_points
from safnet.ops.ball_query import ball_query
from common.nn.functional import batch_index_select
from torch.nn.parameter import Parameter
from safnet.models.attention_method import eca_layer
from safnet.models.sift import PointSIFT_res_module,PointSIFT_module
from safnet.ops.knn_distance import knn_distance

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
            # self.mlp = SharedMLP(in_channels + (10 if use_relation else 0), mlp_channels, ndim=2, bn=True)
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
                # pdb.set_trace()
                diff_xyz = src_xyz[:,:,:,:d] - tgt_xyz.unsqueeze(-1)  # (b, 3, np, k)
                distance = torch.sum(diff_xyz ** 2, dim=1, keepdim=True)  # (b, 1, np, k)
                # weight_distance = torch.nn.functional.softmax(-distance,-1)
                # relation_feature = torch.cat([diff_xyz, distance,src_xyz,tgt_xyz.unsqueeze(-1).repeat(1,1,1,k)], dim=1)
                relation_feature = torch.cat([diff_xyz, distance],
                                             dim=1)
                x = torch.cat([feature[:,:,:,:d], relation_feature], 1)
            else:
                x = feature
            x = self.mlp(x)
            # pdb.set_trace()
            # x = weight_distance * x
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
            # self.mlp1 = SharedMLP(448,(448,128,64))
            # self.mlp2 = SharedMLP(224,(224,128,64))
            # self.mlp3 = SharedMLP(112,(112,64,32))

            # self.pointsift_res_1 = PointSIFT_res_module(radius=0.1, output_channel=64, extra_input_channel=112, merge='add', same_dim=True)
            # self.pointsift_res_2 = PointSIFT_res_module(radius=0.05, output_channel=64, extra_input_channel=56, merge='add', same_dim=True)
            # self.pointsift_res_3 = PointSIFT_res_module(radius=0.03, output_channel=32, extra_input_channel=28, merge='add', same_dim=True)
            # self.factor = 4
            self.pointsift_m1_1 = PointSIFT_module(radius=0.1, output_channel=128, extra_input_channel=192,
                                                        merge='add', same_dim=True)
            # self.pointsift_m2_2 = PointSIFT_module(radius=0.01, output_channel=32, extra_input_channel=int(192/self.factor),
            #                                         merge='add', same_dim=True)
            # self.pointsift_m3 = PointSIFT_module(radius=0.25, output_channel=64, extra_input_channel=48,
            #                                         merge='add', same_dim=True)
            # self.narrow_conv1d = nn.
            # self.conv1d = nn.Conv1d(160,64,1)
            # self.conv1d_1 = nn.Conv1d(67, 64, 1)
            self.conv1d_2 = SharedMLP(128, (64,64 ),
                                       ndim=1, bn=True)
            # self.conv1d2 = nn.Conv1d(128,64,1)
            # self.bn = nn.BatchNorm1d(64)
            self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_1.data.fill_(0.25)
            self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_2.data.fill_(0)
            self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fuse_weight_3.data.fill_(1.0)
            self.linear1 = torch.nn.Linear(64, 64, bias=True)
            # self.mlp2 = SharedMLP(112,(128,128,128))
            self.conv1d_3 = SharedMLP(128, (64,64 ),
                                       ndim=1, bn=True)

            self.pointnet2 = pn2ssg.PN2SSGemb(64)
            # self.dgcnn = DGCNN_semseg(k=20)
            # self.GACNet = GACNet(20)
            # self.pointnet2 = get_model(20)
            # self.pdist = nn.PairwiseDistance(p=2)
            self.linear2 = torch.nn.Linear(128, 128, bias=True)
            # self.dropout = nn.Dropout(p=0.5)
            # self.bn1 = nn.BatchNorm1d(128, affine=False)
            # self.bn3 = nn.BatchNorm1d(128, affine=False)
            # self.bn2 = nn.BatchNorm1d(128, affine=False)
            # self.weight = Parameter(torch.rand([1], requires_grad=True))
        elif self.similarity_mode == 'feature':
            print('woca')
            self.mlp = SharedMLP(224,(128,128,128))
            self.pointnet2 = pn2ssg.PN2SSGemb(128)
            self.weight = Parameter(torch.rand([128,1], requires_grad=True))
    def forward(self, imageneighbors,knnpoints,points,feature_2d3d):
        # pdb.set_trace()
        b,_,n=points.shape
        # print(knnpoints.shape)
        imageneighbors = imageneighbors.permute(0,1,3,2)
        # pdb.set_trace()
        # k = knnpoints.shape[2]

        # small_k = int(knnpoints.shape[2] / self.factor)
        distance = self.geometric_similarity(knnpoints,imageneighbors)

        furthest_image_repre = imageneighbors.reshape(b, -1, n)
        try:
            furthest_real_repre = knnpoints.reshape(b,-1,n)
        except:
            furthest_real_repre = furthest_image_repre
        # int_real_repre = knnpoints[:,:,:small_k,:].reshape(b,-1,n)
        # int_image_repre = imageneighbors[:,:,:small_k,:].reshape(b,-1,n)


        # shallowrealfeat = self.pointsift_res_1(points, furthest_real_repre)
        # shallowimagefeat = self.pointsift_res_1(points, furthest_image_repre)
        # pdb.set_trace()
        shallowimagefeat = self.pointsift_m1_1(points, furthest_image_repre)

        try:
            shallowrealfeat = self.pointsift_m1_1(points, furthest_real_repre)
        except:
            shallowrealfeat = shallowimagefeat
        # shallowrealfeat = torch.cat((self.pointsift_m1_1(points, furthest_real_repre),
        #                              self.pointsift_m2_2(points, int_real_repre)),
        #                             1)
        # furthest_real_repre2 = self.pointsift_m2(points, furthest_real_repre)
        # furthest_real_repre3 = self.pointsift_m3(points, furthest_real_repre)
        # shallowrealfeat = torch.cat((furthest_real_repre1,furthest_real_repre2,furthest_real_repre3),1)

        # shallowimagefeat = torch.cat((self.pointsift_m1_1(points, furthest_image_repre),
        #                              self.pointsift_m2_2(points, int_image_repre)),
        #                             1)
        # furthest_image_repre2 = self.pointsift_m2(points, furthest_image_repre)
        # furthest_image_repre3 = self.pointsift_m3(points, furthest_image_repre)
        # shallowimagefeat = torch.cat((furthest_image_repre1, furthest_image_repre2, furthest_image_repre3), 1)
        # knnpoints = knnpoints.reshape(b,3,-1)
        # imageneighbors = imageneighbors.reshape(b,3,-1)
        # with torch.no_grad():
        #     furthest_index = ball_query(points, knnpoints, self.maxradius, self.max_neighbors) # b,n,max_neighbors
        #     intermediate_index = ball_query(points, knnpoints, self.intradius, self.int_neighbors) # b,n,int_neighbors
        #     nearest_index = ball_query(points, knnpoints, self.minradius, self.min_neighbors) # b, n, min_neighbors
        #     # pdb.set_trace()
        #     furthest_index_img = ball_query(points,imageneighbors,self.maxradius,self.max_neighbors) # b,n,k
        #     intermediate_index_img = ball_query(points, imageneighbors, self.intradius, self.int_neighbors)
        #     nearest_index_img = ball_query(points, imageneighbors, self.minradius, self.min_neighbors)
        #     # pdb.set_trace()
        #     # pdb.set_trace()
        #     furthest_real_repre, furthest_image_repre,realnei1,imgnei1 = self.collect_neighbors(b,n,points,knnpoints,furthest_index,imageneighbors,furthest_index_img,self.max_neighbors) #b,7*max,n
        #     int_real_repre, int_image_repre,realnei2,imgnei2 = self.collect_neighbors(b,n,points,knnpoints,intermediate_index,imageneighbors,intermediate_index_img,self.int_neighbors) #b,7*int,n
        #     nearest_real_repre, nearest_image_repre,realnei3,imgnei3 = self.collect_neighbors(b,n,points,knnpoints,nearest_index,imageneighbors,nearest_index_img,self.min_neighbors) #b,7*min,n
        # pdb.set_trace()

        # shallowrealfeat1 = self.pointsift_res_1(points,furthest_real_repre)
        # shallowrealfeat2 = self.pointsift_res_2(points, int_real_repre)
        # shallowrealfeat3 = self.pointsift_res_3(points, nearest_real_repre)
        # shallowrealfeat = torch.cat((shallowrealfeat1,shallowrealfeat2,shallowrealfeat3),-1).permute(0,2,1)

        # shallowimagefeat1 = self.pointsift_res_1(points, furthest_image_repre)
        # shallowimagefeat2 = self.pointsift_res_2(points, int_image_repre)
        # shallowimagefeat3 = self.pointsift_res_3(points, nearest_image_repre)
        # shallowimagefeat = torch.cat((shallowimagefeat1, shallowimagefeat2, shallowimagefeat3), -1).permute(0, 2, 1)
        # # shallowrealfeat = torch.cat((self.mlp1(furthest_real_repre), self.mlp2(int_real_repre), self.mlp3(nearest_real_repre)), 1)
        # # shallowimagefeat = torch.cat((self.mlp1(furthest_image_repre), self.mlp2(int_image_repre), self.mlp3(nearest_image_repre)), 1) #b,160,n
        #
        # shallowrealfeat = torch.cat((shallowrealfeat,distance.permute(0,2,1)),1)
        # shallowimagefeat = torch.cat((shallowimagefeat,distance.permute(0,2,1)),1)

        shallowrealfeat = self.conv1d_2(shallowrealfeat)
        shallowimagefeat = self.conv1d_2(shallowimagefeat)

        # pdb.set_trace()
        # shallowrealfeat1 = self.linear11(torch.cat((shallowrealfeat.permute(0,2,1),distance),-1)).permute(0,2,1)
        # shallowimagefeat1 = self.linear11(torch.cat((shallowimagefeat.permute(0,2,1),distance),-1)).permute(0,2,1)

        shallowrealfeat1 = self.linear1(shallowrealfeat.permute(0, 2, 1)).permute(0, 2, 1)
        shallowimagefeat1 = self.linear1(shallowimagefeat.permute(0, 2, 1)).permute(0, 2, 1)

        shallow_similarity =  self.cos(shallowrealfeat1,shallowimagefeat1)
        #-------------------------------
        geometric_similarity = torch.exp(-torch.reciprocal(self.fuse_weight_1)*distance[:,:,0])+self.fuse_weight_2
        shallow_similarity = shallow_similarity*geometric_similarity
        feature_2d3d = shallow_similarity.unsqueeze(1).expand(b, 64, n) * feature_2d3d

        shallow_fused_feature = torch.cat((shallowrealfeat,feature_2d3d),1)
        shallow_fused_feature = self.conv1d_3(shallow_fused_feature)
        # pdb.set_trace()
        # -------------------------------
        deeprealfeat = self.pointnet2({'points': points, 'feature': shallow_fused_feature})
        # deepimagefeat = self.pointnet2({'points': points, 'feature': shallowimagefeat})

        # DGCNN
        # deeprealfeat = self.dgcnn({'points': points, 'feature': shallowrealfeat})
        # deepimagefeat = self.dgcnn({'points': points, 'feature': shallowimagefeat})


        # GACNet
        # deeprealfeat = self.GACNet({'points': points, 'feature': shallowrealfeat})
        # deepimagefeat = self.GACNet({'points': points, 'feature': shallowimagefeat})

        # deeprealfeat1 = self.linear2(deeprealfeat.permute(0,2,1)).permute(0,2,1)
        # deepimagefeat1 = self.linear2(deepimagefeat.permute(0,2,1)).permute(0,2,1)

        # realmultiscalefeat = torch.cat([group_points(deeprealfeat, furthest_index),
        #                                 group_points(deeprealfeat, nearest_index),
        #                                 group_points(deeprealfeat, intermediate_index)],-1)
        # imagemultiscalefeat = torch.cat([group_points(deepimagefeat, furthest_index),
        #                                  group_points(deepimagefeat, nearest_index),
        #                                  group_points(deepimagefeat, intermediate_index)],-1)

        if self.similarity_mode == 'point':
            # deep_similarity = self.cos(deeprealfeat1,deepimagefeat1)
            # shallow_similarity = torch.reciprocal(self.pdist(shallowrealfeat, shallowimagefeat))
            # deep_similarity = torch.reciprocal(self.pdist(deeprealfeat, deepimagefeat))
            deep_similarity = torch.ones_like(shallow_similarity) * self.fuse_weight_3
            # pdb.set_trace()
            # print(shallow_similarity,'1111111111111111')
            # print(deep_similarity,'222222222222222222')
            # pdb.set_trace()
            # similarity = deep_similarity + self.weight * shallow_similarity
            return shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat
        elif self.similarity_mode == 'feature':
            # print('woca')
            shallow_similarity = torch.cos(shallowrealfeat-shallowimagefeat)
            deep_similarity = torch.cos(deeprealfeat-deepimagefeat)
            # abc=torch.cosine_similarity(shallowrealfeat, shallowimagefeat, dim=1)
            # pdb.set_trace()
            # similarity = deep_similarity + self.weight * shallow_similarity
            pass
        # pdb.set_trace()
        return shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat
    def collect_neighbors(self, b,n,points,knnpoints,furthest_index,imageneighbors,furthest_index_img,max_neighbors):
        # b = knnpoints.shape[0]
        realneighbors = torch.ones(b,3,n,max_neighbors).cuda()
        imageneighbors2 = torch.ones(b, 3, n, max_neighbors).cuda()
        # for i in range(n):
        #     realneighbors[:,:,i,:] = batch_index_select(points, furthest_index[:, i, :], dim=2)

        # pdb.set_trace()
        for j in range(b):
            realneighbors[j] = knnpoints[j,:,furthest_index[j,:,:]]
            imageneighbors2[j] = imageneighbors[j,:,furthest_index_img[j,:,:]]
        imageneighbors = imageneighbors2
        # points = points.unsqueeze(-1)
        # pdb.set_trace()

        imageneifeat = self.neighbor_feature(imageneighbors,points)
        realneifeat = self.neighbor_feature(realneighbors,points)

        realneirepre = torch.cat([realneighbors, realneifeat], dim=1)     # b,7,n,k
        imageneirepre = torch.cat([imageneighbors, imageneifeat], dim=1)

        realneirepre = torch.reshape(realneirepre,(b,-1,n))
        imageneirepre = torch.reshape(imageneirepre,(b,-1,n))

        return realneirepre,imageneirepre,realneighbors,imageneighbors
    def neighbor_feature(self,neighbors, points):
        # pdb.set_trace()
        diff_xyz = neighbors - points.unsqueeze(-1)
        distance = torch.sum(diff_xyz ** 2, dim=1, keepdim=True)
        relation_feature = torch.cat([diff_xyz, distance], dim=1)
        return relation_feature
    def geometric_similarity(self,knn_realneighbors,knn_imageneighbors):
        # pdb.set_trace()
        a,b,c,d=knn_imageneighbors.shape
        knn_imageneighbors = knn_imageneighbors.permute(0,1,3,2)
        try:
            knn_realneighbors =knn_realneighbors.reshape(a,b,c,d)
            knn_realneighbors = knn_realneighbors.permute(0,1,3,2) #torch.Size([3, 3, 8192, 32])
        except:
            print(a,b,c,d)
            print(knn_realneighbors.shape)
            knn_realneighbors =knn_imageneighbors
        _,distance = knn_distance(knn_realneighbors,knn_imageneighbors,3)
        # pdb.set_trace()
        # index1 = index.reshape(b,n,max_neighbors,3)
        # distance1 = distance.reshape(b, n, max_neighbors,3)
        # local_similarity = 1 - torch.sigmoid(torch.mean(distance1[:,:,:,0],dim=-1))
        # return local_similarity
        return distance
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
        # self.net_3d = net_3d
        # self.net_3d_msg = get_model(20)
        # pdb.set_trace()
        self.similarity = DeepSimilarity(self.similarity_mode)
        self.atten_rgb = self.channel_attention(64)
        self.atten_point = self.channel_attention(128)

        # self.narrow_conv1d = nn.Conv1d(192, 128, 1)
        # self.eca1 = eca_layer(192,1)
        self.mlp_seg = SharedMLPDO(128, (128,128,128), ndim=1, bn=True, p=0.5)
        self.seg_logit = nn.Conv1d(128, 20, 1, bias=True)


        # self.mlp_seg_3d = SharedMLPDO(128, (128,128), ndim=1, bn=True, p=0.5)
        # self.seg_logit_3d = nn.Conv1d(128, 20, 1, bias=True)

        # net_3d_ckpt_path = '0404_no_absolte_coordinate/protect/model_046000.pth'
        # # net_3d_ckpt_path = None
        # if net_3d_ckpt_path:
        #     checkpoint = torch.load(net_3d_ckpt_path, map_location=torch.device("cpu"))
        #     pdb.set_trace()
        #     self.similarity.load_state_dict(checkpoint['model'],strict=False)

        # self.softmax = torch.nn.Softmax(dim=0)
        # self.mlp1 = SharedMLP(192, (192,192),  ndim = 1)
        # self.mlp2 = SharedMLP(192, (192, 192), ndim=1)
        # self.mlp3 = SharedMLP(192, (192, 192), ndim=1)
        # self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

        # self.pointnet2final = pn2ssg.PN2SSG(192,20)

    def forward(self, data_batch):
        # (batch_size, num_views, 3, h, w)
        images = data_batch['images']
        depths = data_batch['depth']
        # labels_2d = data_batch['label_2d']
        b, nv, _, h, w = images.size()
        # collapse first 2 dimensions together
        # pdb.set_trace()
        images = images.reshape([-1] + list(images.shape[2:]))
        depths = depths.reshape([-1] + list(depths.shape[2:]))

        # 2D network
        # preds_2d = self.net_2d({'image': images, 'depth': depths, 'label_2d':data_batch['label_2d']})
        preds_2d = self.net_2d({'image': images, 'depth': depths, 'label_2d':depths})

        feature_2d = preds_2d['feature']  # (b * nv, c, h, w)
        logit_2d = preds_2d['seg_logit']
        # feat_64dim = preds_2d['64dim']
        # feat_128dim = preds_2d['128dim']
        # feat_256dim = preds_2d['256dim']

        # pdb.set_trace()
        # unproject features
        knn_indices = data_batch['knn_indices']  # (b, np, k) most cloes unprojected points
        feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
        feature_2d = feature_2d.reshape(b, -1, nv * h * w)
        # pdb.set_trace()
        feature_2d = group_points(feature_2d, knn_indices[:,:,:3])  # (b, c, np, k)

        # feat_64dim = self.unproject(feat_64dim,2,b,nv,h,w)
        # feat_128dim = self.unproject(feat_128dim, 4, b,nv,h, w)
        # feat_256dim = self.unproject(feat_256dim, 8, b,nv,h, w)

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
        # pdb.set_trace()

        feature_2d3d = self.feat_aggreg(image_xyz, points, feature_2d)
        # pdb.set_trace()
        atten_rgb = self.atten_rgb(feature_2d3d)
        feature_2d3d = feature_2d3d.mul(atten_rgb)

        shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat = self.similarity(image_xyz,knnpoints,points,feature_2d3d) # b,128,n

        # stack_similarity = torch.stack([shallow_similarity,deep_similarity],0)
        # stack_similarity = torch.nn.functional.softmax(stack_similarity, 0)
        # shallow_similarity = stack_similarity[0,:]
        # deep_similarity = stack_similarity[1,:]

        # feature_2d3d = shallow_similarity.unsqueeze(1).expand(b, 64, n) * feature_2d3d
        weighted_deepfeature = deeprealfeat*deep_similarity.unsqueeze(1).expand(b, 128, n)
        # fused_feature = torch.cat([feature_2d3d, weighted_deepfeature], 1)


        atten_point = self.atten_point(weighted_deepfeature)
        weighted_deepfeature = weighted_deepfeature.mul(atten_point)
        
        if self.similarity_mode == 'point':
            # attention
            # pdb.set_trace()
            # fused_feature = self.eca1(fused_feature)
            # fused_feature = torch.nn.functional.relu(self.narrow_conv1d(fused_feature))


            seg_feature = self.mlp_seg(weighted_deepfeature)
            seg_logit = self.seg_logit(seg_feature)

        # elif:
            # Feature Fusion Module

        seg_logit1 = logit_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, nc, nv, h, w)
        seg_logit1 = seg_logit1.reshape(b, -1, nv * h * w)
        # pdb.set_trace()
        seg_logit1 = group_points(seg_logit1, knn_indices[:,:,:1])  # (b, nc, np, k)
        seg_logit1 = seg_logit1.mean(-1)  # (b, nc, np)

        preds = dict()
        # preds['logit_point_branch'] = self.seg_logit_3d(self.mlp_seg_3d(deeprealfeat))
        preds['logit_2d'] = logit_2d
        preds['logit_2d_chunks'] = seg_logit1
        preds['seg_logit'] = seg_logit
        # knn_dist=data_batch['knn_dist']
        # pdb.set_trace()
        return preds,None,preds_2d['depth_loss']

        # pdb.set_trace()
        # 3D network
        # preds_3d = self.pointnet2final({'points': points, 'feature': torch.cat([feature_2d3d,weighted_deepfeature],1)})
        # preds = preds_3d
        # return preds

    def unproject(self, feature_2d, scale_factor,b,nv,h, w):
        newh = int(h/scale_factor)
        neww = int(w/scale_factor)
        # pdb.set_trace()
        feature_2d = feature_2d.reshape(b, nv, -1, newh, neww).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
        feature_2d = feature_2d.reshape(b, -1, nv * newh * neww)
        # feature_2d = group_points(feature_2d, knn_indices)  # (b, c, np, k)
        return feature_2d

    def channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool1d(1)
        conv = nn.Conv1d(num_channel, num_channel, kernel_size=1)
        # bn = nn.BatchNorm2d(num_channel)
        activation = nn.Sigmoid() # todo modify the activation function

        return nn.Sequential(*[pool, conv, activation])
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

