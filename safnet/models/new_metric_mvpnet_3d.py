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
from safnet.ops.fps import farthest_point_sample
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
                diff_xyz = src_xyz[:,:,:,:d] - tgt_xyz.unsqueeze(-1)  # (b, 3, np, k)
                distance = torch.sum(diff_xyz ** 2, dim=1, keepdim=True)  # (b, 1, np, k)
                # weight_distance = torch.nn.functional.softmax(-distance,-1)
                # relation_feature = torch.cat([diff_xyz, distance,src_xyz,tgt_xyz.unsqueeze(-1).repeat(1,1,1,k)], dim=1)
                relation_feature = torch.cat([diff_xyz, distance],
                                             dim=1)
                x = torch.cat([feature[:,:,:,:d], relation_feature], 1)
            else:
                x = feature
            # pdb.set_trace()
            x = self.mlp(x)
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
    def __init__(self,similarity_mode,radius=0.08,max_neighbors=64):
        super(DeepSimilarity, self).__init__()
        self.similarity_mode = similarity_mode
        self.maxradius = radius
        self.intradius = radius/2
        self.minradius = radius/4
        # self.minminradius
        self.max_neighbors = max_neighbors
        self.int_neighbors = int(max_neighbors/2)
        self.min_neighbors = int(max_neighbors/4)
        if self.similarity_mode=='point':
            self.cos = torch.nn.CosineSimilarity(dim=1)
            self.mlp1 = SharedMLP(448,(448,128,64))
            self.mlp2 = SharedMLP(224,(224,128,64))
            self.mlp3 = SharedMLP(112,(112,64,32))

            # self.newmlp1 = SharedMLP(256,(256,128,64),ndim=1)
            # self.newmlp2 = SharedMLP(128,(128,64,32),ndim=1)
            # self.newmlp3 = SharedMLP(64,(64,32,16),ndim=1)

            # self.newmlp1 = SharedMLP(4,(64,64),ndim=2)
            # self.newmlp2 = SharedMLP(4,(32,32),ndim=2)
            # self.newmlp3 = SharedMLP(4,(16,16),ndim=2)
            # if is_training:
            # self.decoder1 = pn2ssg.FoldingNet(64, num_neighbors=self.max_neighbors)
            # self.decoder2 = pn2ssg.FoldingNet(64, num_neighbors=self.int_neighbors)
            # self.decoder3 = pn2ssg.FoldingNet(32, num_neighbors=self.min_neighbors)

            # self.maxpooling1 = nn.AdaptiveMaxPool2d((None,1))
            # self.maxpooling2 = nn.AdaptiveMaxPool2d((None, 1))
            # self.maxpooling3 = nn.AdaptiveMaxPool2d((None, 1))

            self.conv1d = nn.Conv1d(160,64,1)

            self.linear1 = torch.nn.Linear(64, 64, bias=True)

            self.pointnet2 = pn2ssg.PN2SSGemb2(64)

            self.linear2 = torch.nn.Linear(128, 128, bias=True)

        elif self.similarity_mode == 'feature':
            print('woca')
            self.mlp = SharedMLP(224,(128,128,128))
            self.pointnet2 = pn2ssg.PN2SSGemb(128)
            self.weight = Parameter(torch.rand([128,1], requires_grad=True))
    def forward(self, imageneighbors,knnpoints,points,is_training):
        b,_,n=points.shape
        # print(knnpoints.shape)
        # imageneighbors = points
        imageneighbors = imageneighbors.reshape(b,3,-1)
        with torch.no_grad():
            furthest_index = ball_query(points, knnpoints, self.maxradius, self.max_neighbors) # b,n,max_neighbors
            intermediate_index = ball_query(points, knnpoints, self.intradius, self.int_neighbors) # b,n,int_neighbors
            nearest_index = ball_query(points, knnpoints, self.minradius, self.min_neighbors) # b, n, min_neighbors
            # pdb.set_trace()
            furthest_index_img = ball_query(points,imageneighbors,self.maxradius,self.max_neighbors) # b,n,k
            intermediate_index_img = ball_query(points, imageneighbors, self.intradius, self.int_neighbors)
            nearest_index_img = ball_query(points, imageneighbors, self.minradius, self.min_neighbors)
            # pdb.set_trace()
            # pdb.set_trace()
            # realneifeat, imageneifeat, realneirepre, imageneirepre
            local_similarity1,realneifeat1, imageneifeat1,furthest_real_repre, furthest_image_repre = self.collect_neighbors(b,n,points,knnpoints,furthest_index,imageneighbors,furthest_index_img,self.max_neighbors) #b,7*max,n
            local_similarity2,realneifeat2, imageneifeat2,int_real_repre, int_image_repre = self.collect_neighbors(b,n,points,knnpoints,intermediate_index,imageneighbors,intermediate_index_img,self.int_neighbors) #b,7*int,n
            local_similarity3,realneifeat3, imageneifeat3,nearest_real_repre, nearest_image_repre = self.collect_neighbors(b,n,points,knnpoints,nearest_index,imageneighbors,nearest_index_img,self.min_neighbors) #b,7*min,n
            local_similarity = local_similarity1
            # local_similarity = (local_similarity1+local_similarity2+local_similarity3)/3
        # shallowrealfeat = torch.cat((self.newmlp1(furthest_real_repre), self.newmlp2(int_real_repre), self.newmlp3(nearest_real_repre)), 1)
        # shallowimagefeat = torch.cat((self.newmlp1(furthest_image_repre), self.newmlp2(int_image_repre), self.newmlp3(nearest_image_repre)), 1) #b,160,n

        shallowrealfeat1 = self.mlp1(furthest_real_repre)
        # shallowimagefeat1 = self.mlp1(furthest_image_repre)

        shallowrealfeat2 = self.mlp2(int_real_repre)
        # shallowimagefeat2 = self.mlp2(int_image_repre)

        shallowrealfeat3 = self.mlp3(nearest_real_repre)
        # shallowimagefeat3 = self.mlp3(nearest_image_repre)

        if is_training:
        # pdb.set_trace()
            with torch.no_grad():
                # from safnet.ops.fps import farthest_point_sample
                # index2recon = farthest_point_sample(points, 32)  # (batch_size, num_centroids)
                b = points.shape[0]
                index2recon = torch.from_numpy(np.random.randint(0,8192,size=(b,32))).cuda()
                # points2recon = batch_index_select(points, index2recon, dim=2)  # (batch_size, 3, num_centroids)
                sl_real_feat2recon1 = batch_index_select(shallowrealfeat1, index2recon, dim=2)
                # sl_img_feat2recon1 = batch_index_select(shallowimagefeat1, index2recon, dim=2)
                sl_real_feat2recon2 = batch_index_select(shallowrealfeat2, index2recon, dim=2)
                # sl_img_feat2recon2 = batch_index_select(shallowimagefeat2, index2recon, dim=2)
                sl_real_feat2recon3 = batch_index_select(shallowrealfeat3, index2recon, dim=2)
                # sl_img_feat2recon3 = batch_index_select(shallowimagefeat3, index2recon, dim=2)

                realneighbors2recon1 = batch_index_select(realneifeat1[:, :3, :, :], index2recon, dim=2)
                # imgneighbors2recon1 = batch_index_select(imageneifeat1[:, :3, :, :], index2recon, dim=2)
                realneighbors2recon2 = batch_index_select(realneifeat2[:, :3, :, :], index2recon, dim=2)
                # imgneighbors2recon2 = batch_index_select(imageneifeat2[:, :3, :, :], index2recon, dim=2)
                realneighbors2recon3 = batch_index_select(realneifeat3[:, :3, :, :], index2recon, dim=2)
                # imgneighbors2recon3 = batch_index_select(imageneifeat3[:, :3, :, :], index2recon, dim=2)
                # return realneighbors2recon, imgneighbors2recon
                realneighbors2recon = [realneighbors2recon1,realneighbors2recon2,realneighbors2recon3]
                # imgneighbors2recon = [imgneighbors2recon1,imgneighbors2recon2,imgneighbors2recon3]
        # pdb.set_trace()
        # if is_training:
            real_recon1 = self.decoder1(sl_real_feat2recon1)
            # image_recon1 = self.decoder1(sl_img_feat2recon1)
            real_recon2 = self.decoder2(sl_real_feat2recon2)
            # image_recon2 = self.decoder2(sl_img_feat2recon2)
            real_recon3 = self.decoder3(sl_real_feat2recon3)
            # image_recon3 = self.decoder3(sl_img_feat2recon3)
            # return real_recon,image_recon
            real_recon = [real_recon1,real_recon2,real_recon3]
            # image_recon = [image_recon1,image_recon2,image_recon3]

        # shallowrealfeat1 = self.maxpooling1(shallowrealfeat1).squeeze(-1)
        # shallowimagefeat1 = self.maxpooling1(shallowimagefeat1).squeeze(-1)
        #
        # shallowrealfeat2 = self.maxpooling2(shallowrealfeat2).squeeze(-1)
        # shallowimagefeat2 = self.maxpooling2(shallowimagefeat2).squeeze(-1)
        #
        # shallowrealfeat3 = self.maxpooling3(shallowrealfeat3).squeeze(-1)
        # shallowimagefeat3 = self.maxpooling3(shallowimagefeat3).squeeze(-1)

        # pdb.set_trace()
        shallowrealfeat = torch.cat([shallowrealfeat1,shallowrealfeat2,shallowrealfeat3],dim=1)
        # shallowimagefeat = torch.cat([shallowimagefeat1,shallowimagefeat2,shallowimagefeat3],dim=1)

        shallowrealfeat = self.conv1d(shallowrealfeat)
        # shallowimagefeat = self.conv1d(shallowimagefeat)

        # shallowrealfeat1 = self.linear1(shallowrealfeat.permute(0,2,1)).permute(0,2,1)
        # shallowimagefeat1 = self.linear1(shallowimagefeat.permute(0,2,1)).permute(0,2,1)

        deeprealfeat = self.pointnet2({'points': points, 'feature': shallowrealfeat})
        # deepimagefeat = self.pointnet2({'points': points, 'feature': shallowimagefeat})

        # GACNet
        # deeprealfeat = self.GACNet({'points': points, 'feature': shallowrealfeat})
        # deepimagefeat = self.GACNet({'points': points, 'feature': shallowimagefeat})

        # deeprealfeat1 = self.linear2(deeprealfeat.permute(0,2,1)).permute(0,2,1)
        # deepimagefeat1 = self.linear2(deepimagefeat.permute(0,2,1)).permute(0,2,1)

        # pdb.set_trace()
        # realmultiscalefeat = torch.cat([group_points(deeprealfeat, furthest_index),
        #                                 group_points(deeprealfeat, nearest_index),
        #                                 group_points(deeprealfeat, intermediate_index)],-1)
        # imagemultiscalefeat = torch.cat([group_points(deepimagefeat, furthest_index),
        #                                  group_points(deepimagefeat, nearest_index),
        #                                  group_points(deepimagefeat, intermediate_index)],-1)

        if self.similarity_mode == 'point':
            # shallow_similarity =  self.cos(shallowrealfeat1,shallowimagefeat1)
            # deep_similarity = self.cos(deeprealfeat1,deepimagefeat1)

            shallow_similarity = local_similarity * 2
            deep_similarity = torch.ones_like(local_similarity)
            # shallow_similarity = torch.ones_like(shallow_similarity)
            # deep_similarity = torch.ones_like(deep_similarity)
            # pdist = nn.PairwiseDistance(p=2)
            # shallow_similarity = pdist(shallowrealfeat, shallowimagefeat)
            if is_training:
                return shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat, real_recon, None, realneighbors2recon, None
            else:
                return shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat, None, None, None, None

            # return shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat,real_recon,image_recon,realneighbors2recon,imgneighbors2recon
            # deep_similarity = pdist(deeprealfeat1, deepimagefeat1)
            # pdb.set_trace()

        elif self.similarity_mode == 'feature':
            # print('woca')
            shallow_similarity = torch.cos(shallowrealfeat-shallowimagefeat)
            deep_similarity = torch.cos(deeprealfeat-deepimagefeat)

            # abc=torch.cosine_similarity(shallowrealfeat, shallowimagefeat, dim=1)
            # pdb.set_trace()
            # similarity = deep_similarity + self.weight * shallow_similarity
            pass
        # pdb.set_trace()
        return shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat,realneighbors,imageneighbors
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

        knn_realneighbors = realneighbors.reshape(b*n,3,max_neighbors)
        knn_imageneighbors = imageneighbors.reshape(b*n,3,max_neighbors)
        # # index, distance = knn_distance(knn_realneighbors,knn_imageneighbors,3)
        # knn_points = points.reshape(b*n,1,3)
        # image_centroid = knn_imageneighbors - knn_points
        # image_centroid_dist = torch.sqrt(torch.sum(image_centroid ** 2, -1))
        #
        # real_centroid = knn_realneighbors - knn_points
        # real_centroid_dist = torch.sqrt(torch.sum(real_centroid ** 2, -1))
        # for i in range(b*n):
        #     num_out_ball = len(image_centroid_dist[i, image_centroid_dist[i] > real_centroid_dist.max()])
        #     # out_ball_tensor = torch.zeros_like(knn_points)
        #     if num_out_ball == max_neighbors:
        #         pass
        #     if num_out_ball != 0:
        #         # pdb.set_trace()
        #         print(num_out_ball)
        #         knn_imageneighbors[i, image_centroid_dist[i] > real_centroid_dist.max()] = knn_points[i]
        # pdb.set_trace()
        index,distance = knn_distance(knn_realneighbors,knn_imageneighbors,3)
        index1 = index.reshape(b,n,max_neighbors,3)
        distance1 = distance.reshape(b, n, max_neighbors,3)
        local_similarity = 1 - torch.sigmoid(torch.mean(distance1[:,:,:,0],dim=-1))
        # pdb.set_trace()

        imageneifeat = self.neighbor_feature(imageneighbors,points)
        realneifeat = self.neighbor_feature(realneighbors,points)

        # imageneirepre = imageneifeat
        # realneirepre = realneifeat
        # pdb.set_trace()



        # realneirepre = torch.reshape(realneifeat.permute(0,1,3,2),(b,-1,n))
        # imageneirepre = torch.reshape(imageneifeat.permute(0,1,3,2),(b,-1,n))

        realneirepre = torch.cat([realneighbors, realneifeat], dim=1)     # b,7,n,k
        imageneirepre = torch.cat([imageneighbors, imageneifeat], dim=1)

        realneirepre = torch.reshape(realneirepre,(b,-1,n))
        imageneirepre = torch.reshape(imageneirepre,(b,-1,n))



        return local_similarity,realneifeat,imageneifeat,realneirepre,imageneirepre
    def neighbor_feature(self,neighbors, points):
        # pdb.set_trace()
        diff_xyz = neighbors - points.unsqueeze(-1)
        # pdb.set_trace()
        distance = torch.sum(diff_xyz ** 2, dim=1, keepdim=True)
        relation_feature = torch.cat([diff_xyz, distance], dim=1)
        return relation_feature

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
        self.similarity = DeepSimilarity(self.similarity_mode)
        self.narrow_conv1d = nn.Conv1d(192, 128, 1)
        self.mlp_seg = SharedMLPDO(128, (128,128,128), ndim=1, bn=True, p=0.5)
        self.seg_logit = nn.Conv1d(128, 20, 1, bias=True)
        self.chamfer = ChamferLoss()
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

    def forward(self, data_batch,is_training):
        # (batch_size, num_views, 3, h, w)
        images = data_batch['images']
        depths = data_batch['depth']
        b, nv, _, h, w = images.size()
        # collapse first 2 dimensions together
        images = images.reshape([-1] + list(images.shape[2:]))
        # pdb.set_trace()
        depths = depths.reshape([-1] + list(depths.shape[2:]))
        # 2D network
        preds_2d = self.net_2d({'image': images, 'depth': depths})
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
        feature_2d = group_points(feature_2d, knn_indices)  # (b, c, np, k)

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
        b,_,n = points.shape

        feature_2d3d = self.feat_aggreg(image_xyz, points, feature_2d)

        shallow_similarity, deep_similarity, shallowrealfeat, deeprealfeat,real_recon,image_recon,realneighbors,imageneighbors = self.similarity(image_xyz,knnpoints,points,is_training) # b,128,n

        # pdb.set_trace()
        stack_similarity = torch.stack([shallow_similarity,deep_similarity],0)
        stack_similarity = torch.nn.functional.softmax(stack_similarity, 0)
        shallow_similarity = stack_similarity[0,:]
        deep_similarity = stack_similarity[1,:]

        feature_2d3d = shallow_similarity.unsqueeze(1).expand(b, 64, n) * feature_2d3d
        weighted_deepfeature = deeprealfeat*deep_similarity.unsqueeze(1).expand(b, 128, n)
        fused_feature = torch.cat([feature_2d3d, weighted_deepfeature], 1)

        if self.similarity_mode == 'point':
            fused_feature = torch.nn.functional.relu(self.narrow_conv1d(fused_feature))


            seg_feature = self.mlp_seg(fused_feature)
            seg_logit = self.seg_logit(seg_feature)

        # elif:
            # Feature Fusion Module

        seg_logit1 = logit_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, nc, nv, h, w)
        seg_logit1 = seg_logit1.reshape(b, -1, nv * h * w)
        seg_logit1 = group_points(seg_logit1, knn_indices)  # (b, nc, np, k)
        seg_logit1 = seg_logit1.mean(-1)  # (b, nc, np)

        preds = dict()
        preds['logit_2d'] = logit_2d
        preds['logit_2d_chunks'] = seg_logit1
        preds['seg_logit'] = seg_logit
        # real_recon, image_recon, realneighbors, imageneighbors
        # preds['real_recon'] = real_recon
        # preds['image_recon'] = image_recon
        # preds['realneighbors'] = realneighbors
        # preds['imageneighbors'] = imageneighbors
        # pdb.set_trace()
        if is_training:
            recon_loss = 0
            for i in range(len(real_recon)):
                real_recon1 = real_recon[i].view(b,3,-1).permute(0,2,1)
                # image_recon1 = image_recon[i].view(b,3,-1).permute(0,2,1)
                realneighbors1 = realneighbors[i].view(b,3,-1).permute(0,2,1)
                # imageneighbors1 = imageneighbors[i].view(b,3,-1).permute(0,2,1)
                # pdb.set_trace()

                recon_loss = recon_loss + self.chamfer(real_recon1,realneighbors1) #+ \
                                  # self.chamfer(image_recon1, imageneighbors1)
            # loss_dict['recon_loss'] = recon_loss
            # pdb.set_trace()
            return preds, recon_loss/len(real_recon),preds_2d['mse_loss']
        else:
            return preds, None,preds_2d['mse_loss']
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

    def get_loss(self, cfg):
        from safnet.models.loss import SegLoss, OhemCELoss
        if cfg.TRAIN.LABEL_WEIGHTS_PATH:
            weights = np.loadtxt(cfg.TRAIN.LABEL_WEIGHTS_PATH, dtype=np.float32)
            weights = torch.from_numpy(weights).cuda()
        else:
            weights = None
        return SegLoss(weight=weights)
        # return OhemCELoss(0.7, n_min=8192, weight=weights,ignore_index=-100)

    def get_metric(self, cfg):
        from safnet.models.metric import SegAccuracy, SegIoU
        metric_fn = lambda: [SegAccuracy(), SegIoU(20)]
        return metric_fn(), metric_fn()

class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        '''
        :param x: (bs, np, 3)
        :param y: (bs, np, 3)
        :return: loss
        '''

        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        dist = torch.sqrt(1e-6 + torch.sum(torch.pow(x - y, 2), 3)) # bs, ny, nx
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)

        return min1.mean() + min2.mean()

