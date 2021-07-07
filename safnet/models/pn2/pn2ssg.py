"""PointNet2(single-scale grouping)

References:
    @article{qi2017pointnetplusplus,
      title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
      author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1706.02413},
      year={2017}
    }

"""

import numpy as np
import torch
import torch.nn as nn
import pdb

from common.nn import SharedMLP,SharedMLPDO
from common.nn.init import xavier_uniform
from safnet.models.pn2.modules import SetAbstraction, FeaturePropagation
from safnet.models.pn2.pointconv_util import PointConvDensitySetAbstraction
from safnet.ops.group_points import group_points
import sys
# sys.path.append('/home/syy/zlq/Pointnet_Pointnet2_pytorch/models')
# from pointnet_util import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class PN2SSG(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
                 num_centroids=(2048, 512, 128, 32),
                 radius=(0.1, 0.2, 0.4, 0.8),
                 max_neighbors=(32, 32, 32, 32),
                 fp_channels=((256, 256), (256, 256), (256, 128), (128, 128, 128)),
                 fp_neighbors=(3, 3, 3, 3),
                 seg_channels=(128,),
                 dropout_prob=0.5,
                 use_xyz=True):
        super(PN2SSG, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_xyz = use_xyz

        # sanity check
        num_sa_layers = len(sa_channels)
        num_fp_layers = len(fp_channels)
        assert len(num_centroids) == num_sa_layers
        assert len(radius) == num_sa_layers
        assert len(max_neighbors) == num_sa_layers
        assert num_sa_layers == num_fp_layers
        assert len(fp_neighbors) == num_fp_layers

        # Set Abstraction Layers
        c_in = in_channels
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = SetAbstraction(in_channels=c_in,
                                       mlp_channels=sa_channels[ind],
                                       num_centroids=num_centroids[ind],
                                       radius=radius[ind],
                                       max_neighbors=max_neighbors[ind],
                                       use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            c_in = sa_channels[ind][-1]

        # Get channels for all the intermediate features
        # Ignore the input feature
        # feature_channels = [self.in_channels]
        feature_channels = [0]
        feature_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        c_in = feature_channels[-1]
        self.fp_modules = nn.ModuleList()
        for ind in range(num_fp_layers):
            fp_module = FeaturePropagation(in_channels=c_in,
                                           in_channels_prev=feature_channels[-2 - ind],
                                           mlp_channels=fp_channels[ind],
                                           num_neighbors=fp_neighbors[ind])
            self.fp_modules.append(fp_module)
            c_in = fp_channels[ind][-1]

        # MLP
        self.mlp_seg = SharedMLPDO(fp_channels[-1][-1], seg_channels, ndim=1, bn=True, p=dropout_prob)
        self.seg_logit = nn.Conv1d(seg_channels[-1], num_classes, 1, bias=True)

        # Initialize
        self.reset_parameters()

    def forward(self, data_batch):
        xyz = data_batch['points']
        feature = data_batch.get('feature', None)
        preds = dict()

        xyz_list = [xyz]
        # sa_feature_list = [feature]
        sa_feature_list = [None]

        # Set Abstraction Layers
        for sa_ind, sa_module in enumerate(self.sa_modules):
            xyz, feature = sa_module(xyz, feature)
            xyz_list.append(xyz)
            sa_feature_list.append(feature)

        # Feature Propagation Layers
        fp_feature_list = []
        for fp_ind, fp_module in enumerate(self.fp_modules):
            fp_feature = fp_module(
                xyz_list[-2 - fp_ind],
                xyz_list[-1 - fp_ind],
                sa_feature_list[-2 - fp_ind],
                fp_feature_list[-1] if len(fp_feature_list) > 0 else sa_feature_list[-1],
            )
            fp_feature_list.append(fp_feature)

        # MLP
        seg_feature = self.mlp_seg(fp_feature_list[-1])
        seg_logit = self.seg_logit(seg_feature)

        preds['seg_logit'] = seg_logit
        return preds

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)

    def get_loss(self, cfg):
        from safnet.models.loss import SegLoss
        if cfg.TRAIN.LABEL_WEIGHTS_PATH:
            weights = np.loadtxt(cfg.TRAIN.LABEL_WEIGHTS_PATH, dtype=np.float32)
            weights = torch.from_numpy(weights).cuda()
        else:
            weights = None
        return SegLoss(weight=weights)

    def get_metric(self, cfg):
        from safnet.models.metric import SegAccuracy, SegIoU
        metric_fn = lambda: [SegAccuracy(), SegIoU(self.num_classes)]
        return metric_fn(), metric_fn()

class PN2SSGemb(nn.Module):
    def __init__(self,
                 in_channels,
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
                 num_centroids=(2048, 512, 128, 32),
                 radius=(0.1, 0.2, 0.4, 0.8),
                 max_neighbors=(32, 32, 32, 32),
                 fp_channels=((256, 256), (256, 256), (256, 128), (128, 128, 128)),
                 fp_neighbors=(3, 3, 3, 3),
                 use_xyz=True):
        super(PN2SSGemb, self).__init__()

        self.in_channels = in_channels
        self.use_xyz = use_xyz

        # sanity check
        num_sa_layers = len(sa_channels)
        num_fp_layers = len(fp_channels)
        assert len(num_centroids) == num_sa_layers
        assert len(radius) == num_sa_layers
        assert len(max_neighbors) == num_sa_layers
        assert num_sa_layers == num_fp_layers
        assert len(fp_neighbors) == num_fp_layers

        # Set Abstraction Layers
        c_in = in_channels
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = SetAbstraction(in_channels=c_in,
                                       mlp_channels=sa_channels[ind],
                                       num_centroids=num_centroids[ind],
                                       radius=radius[ind],
                                       max_neighbors=max_neighbors[ind],
                                       use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            c_in = sa_channels[ind][-1]

        # Get channels for all the intermediate features
        # Ignore the input feature
        # feature_channels = [self.in_channels]
        feature_channels = [0]
        feature_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        c_in = feature_channels[-1]
        self.fp_modules = nn.ModuleList()
        for ind in range(num_fp_layers):
            fp_module = FeaturePropagation(in_channels=c_in,
                                           in_channels_prev=feature_channels[-2 - ind],
                                           mlp_channels=fp_channels[ind],
                                           num_neighbors=fp_neighbors[ind])
            self.fp_modules.append(fp_module)
            c_in = fp_channels[ind][-1]

        # Initialize
        self.reset_parameters()

    def forward(self, data_batch):
        xyz = data_batch['points']
        feature = data_batch.get('feature', None)
        preds = dict()

        xyz_list = [xyz]
        # sa_feature_list = [feature]
        sa_feature_list = [None]

        # Set Abstraction Layers
        for sa_ind, sa_module in enumerate(self.sa_modules):
            xyz, feature = sa_module(xyz, feature)
            xyz_list.append(xyz)
            sa_feature_list.append(feature)

        # Feature Propagation Layers
        fp_feature_list = []
        for fp_ind, fp_module in enumerate(self.fp_modules):
            fp_feature = fp_module(
                xyz_list[-2 - fp_ind],
                xyz_list[-1 - fp_ind],
                sa_feature_list[-2 - fp_ind],
                fp_feature_list[-1] if len(fp_feature_list) > 0 else sa_feature_list[-1],
            )
            fp_feature_list.append(fp_feature)


        return fp_feature_list[-1]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)


class PointConv(nn.Module):
    def __init__(self,
                 in_channels,
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
                 num_centroids=(2048, 512, 128, 32),
                 radius=(0.1, 0.2, 0.4, 0.8),
                 max_neighbors=(32, 32, 32, 32),
                 fp_channels=((256, 256), (256, 256), (256, 128), (128, 128, 128)),
                 fp_neighbors=(3, 3, 3, 3),
                 use_xyz=True):
        super(PointConv, self).__init__()

        self.in_channels = in_channels
        self.use_xyz = use_xyz

        # sanity check
        num_sa_layers = len(sa_channels)
        num_fp_layers = len(fp_channels)
        assert len(num_centroids) == num_sa_layers
        assert len(radius) == num_sa_layers
        assert len(max_neighbors) == num_sa_layers
        assert num_sa_layers == num_fp_layers
        assert len(fp_neighbors) == num_fp_layers

        # Set Abstraction Layers
        c_in = in_channels
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = PointConvDensitySetAbstraction(in_channel=c_in+3,
                                       mlp=sa_channels[ind],
                                       npoint=num_centroids[ind],
                                       bandwidth=radius[ind],
                                       nsample=max_neighbors[ind],
                                       group_all=use_xyz)
            # in_channel, mlp, npoint, bandwidth, nsample, group_all
            self.sa_modules.append(sa_module)
            c_in = sa_channels[ind][-1]

        # Get channels for all the intermediate features
        # Ignore the input feature
        # feature_channels = [self.in_channels]
        feature_channels = [0]
        feature_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        c_in = feature_channels[-1]
        self.fp_modules = nn.ModuleList()
        for ind in range(num_fp_layers):
            fp_module = FeaturePropagation(in_channels=c_in,
                                           in_channels_prev=feature_channels[-2 - ind],
                                           mlp_channels=fp_channels[ind],
                                           num_neighbors=fp_neighbors[ind])
            self.fp_modules.append(fp_module)
            c_in = fp_channels[ind][-1]

        # Initialize
        self.reset_parameters()

    def forward(self, data_batch):
        xyz = data_batch['points']
        feature = data_batch.get('feature', None)
        preds = dict()

        xyz_list = [xyz]
        # sa_feature_list = [feature]
        sa_feature_list = [None]

        # Set Abstraction Layers
        for sa_ind, sa_module in enumerate(self.sa_modules):
            xyz, feature = sa_module(xyz, feature)
            xyz_list.append(xyz)
            sa_feature_list.append(feature)

        # Feature Propagation Layers
        fp_feature_list = []
        for fp_ind, fp_module in enumerate(self.fp_modules):
            fp_feature = fp_module(
                xyz_list[-2 - fp_ind],
                xyz_list[-1 - fp_ind],
                sa_feature_list[-2 - fp_ind],
                fp_feature_list[-1] if len(fp_feature_list) > 0 else sa_feature_list[-1],
            )
            fp_feature_list.append(fp_feature)


        return fp_feature_list[-1]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)


class PN2SSGemb2(nn.Module):
    def __init__(self,
                 in_channels,
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
                 num_centroids=(2048, 512, 128, 32),
                 radius=(0.1, 0.2, 0.4, 0.8),
                 max_neighbors=(32, 32, 32, 32),
                 fp_channels=((256, 256), (256, 256), (256, 128), (128, 128, 128)),
                 fp_neighbors=(3, 3, 3, 3),
                 use_xyz=True):
        super(PN2SSGemb2, self).__init__()

        self.in_channels = in_channels
        self.use_xyz = use_xyz

        # sanity check
        num_sa_layers = len(sa_channels)
        num_fp_layers = len(fp_channels)
        assert len(num_centroids) == num_sa_layers
        assert len(radius) == num_sa_layers
        assert len(max_neighbors) == num_sa_layers
        assert num_sa_layers == num_fp_layers
        assert len(fp_neighbors) == num_fp_layers

        # Set Abstraction Layers
        c_in = in_channels
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = SetAbstraction(in_channels=c_in,
                                       mlp_channels=sa_channels[ind],
                                       num_centroids=num_centroids[ind],
                                       radius=radius[ind],
                                       max_neighbors=max_neighbors[ind],
                                       use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            c_in = sa_channels[ind][-1]

        # Get channels for all the intermediate features
        # Ignore the input feature
        # feature_channels = [self.in_channels]
        feature_channels = [0]
        feature_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        c_in = feature_channels[-1]
        self.fp_modules = nn.ModuleList()
        for ind in range(num_fp_layers):
            fp_module = FeaturePropagation(in_channels=c_in,
                                           in_channels_prev=feature_channels[-2 - ind],
                                           mlp_channels=fp_channels[ind],
                                           num_neighbors=fp_neighbors[ind])
            self.fp_modules.append(fp_module)
            c_in = fp_channels[ind][-1]

        # Initialize
        self.reset_parameters()

    def forward(self, data_batch):
        xyz = data_batch['points']
        feature = data_batch.get('feature', None)
        preds = dict()

        xyz_list = [xyz]
        # sa_feature_list = [feature]
        sa_feature_list = [None]

        subindice = []
        # Set Abstraction Layers
        for sa_ind, sa_module in enumerate(self.sa_modules):
            xyz, feature = sa_module(xyz, feature)
            xyz_list.append(xyz)
            sa_feature_list.append(feature)
            # subindice.append(indice)


        # batch_size = knn_indices.shape[0]
        # pointindice0 = torch.ones(batch_size,subindice[0].shape[-1]).to(feat_64dim.device) #b,2048
        # pointindice1 = torch.ones(batch_size, subindice[1].shape[-1]).to(feat_128dim.device) #b,512
        # pointindice2 = torch.ones(batch_size, subindice[2].shape[-1]).to(feat_256dim.device) #b,128
        #
        # for i in range(batch_size):
        #     pointindice0[i,:] = knn_indices[i,subindice[0][i,:],0]
        #     pointindice1[i,:] = pointindice0[i,:][subindice[1][i,:]]
        #     pointindice2[i, :] = pointindice1[i, :][subindice[2][i, :]]
        #
        #
        # pointindice0 = torch.ceil((pointindice0 + 1) / 4 - 0.5) - 1
        # pointindice1 = torch.ceil((pointindice1 + 1) / 16 - 0.5) - 1
        # pointindice2 = torch.ceil((pointindice2 + 1) / 64 - 0.5) - 1
        #
        # pointindice0[pointindice0 < 0] = 0
        # pointindice1[pointindice1 < 0] = 0
        # pointindice2[pointindice2 < 0] = 0
        #
        # # if torch.max(pointindice0.unsqueeze(-1).long()) >14399 or torch.min(pointindice0.unsqueeze(-1).long()) < 0:
        # #     pdb.set_trace()
        #
        # # with torch.no_grad():
        # feat_64dim = group_points(feat_64dim, pointindice0.unsqueeze(-1).long()).squeeze(-1)
        # feat_128dim = group_points(feat_128dim, pointindice1.unsqueeze(-1).long()).squeeze(-1)
        # feat_256dim = group_points(feat_256dim, pointindice2.unsqueeze(-1).long()).squeeze(-1)

        # pdb.set_trace()


        # Feature Propagation Layers
        fp_feature_list = []
        for fp_ind, fp_module in enumerate(self.fp_modules):
            # pdb.set_trace()
            fp_feature = fp_module(
                xyz_list[-2 - fp_ind],
                xyz_list[-1 - fp_ind],
                sa_feature_list[-2 - fp_ind],
                fp_feature_list[-1] if len(fp_feature_list) > 0 else sa_feature_list[-1]
            )
            fp_feature_list.append(fp_feature)


        return fp_feature_list[-1]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)





class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(2048, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(32, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x, l4_points

class FoldingNet(nn.Module):
    def __init__(self, in_channel, num_neighbors):
        super().__init__()

        self.in_channel = in_channel
        self.num_neighbors = num_neighbors

        a = torch.linspace(-1., 1., steps=32, dtype=torch.float).view(1, 32).expand(1, 32).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=32, dtype=torch.float).view(32, 1).expand(32, 1).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0)


        self.folding1 = nn.Sequential(
            # nn.Conv1d(in_channel + 2, 64, 1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
            SharedMLP(in_channel + 2,(int(in_channel/2),int(in_channel/4)), ndim=2, bn=True),
            # nn.Conv1d(64, 32, 1),
            # nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel/4),3,1),
        )

        self.folding2 = nn.Sequential(
            # nn.Conv1d(in_channel + 3, 64, 1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
            SharedMLP(in_channel + 3, (int(in_channel/2), int(in_channel/4)), ndim=2, bn=True),
            # nn.Conv1d(64, 32, 1),
            # nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel/4), 3, 1),
        )

    def forward(self, x):
        bs = x.size(0)
        # c = x.size(1)
        n = x.size(2)
        x = x.view(bs,self.in_channel,n,1).expand(bs,self.in_channel,n,self.num_neighbors)
        k = x.size(3)
        # features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, 1024)
        # pdb.set_trace()
        features = x.view(bs,self.in_channel,n,k)
        seed = self.folding_seed.view(1, 2, n, 1).expand(bs, 2, n, k).to(x.device)

        x = torch.cat([seed, features], dim=1)

        # pdb.set_trace()
        fd1 = self.folding1(x)
        # pdb.set_trace()
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        # pdb.set_trace()

        return fd2

def test(b=2, c=0, n=8192):
    data_batch = dict()
    data_batch['points'] = torch.randn(b, 3, n)
    if c > 0:
        data_batch['feature'] = torch.randn(b, c, n)
    data_batch = {k: v.cuda() for k, v in data_batch.items()}

    net = PN2SSG(c, 20)
    net = net.cuda()
    print(net)
    preds = net(data_batch)
    for k, v in preds.items():
        print(k, v.shape)

if __name__=="__main__":
    test()