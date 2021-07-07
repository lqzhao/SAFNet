import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import models.extractors as extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNetResNet34(nn.Module):
# class PSPNet(nn.Module):
    def __init__(self, num_classes=20, p = 0.0,sizes = (1, 2, 3, 6), psp_size = 512, deep_features_size = 256, backend = 'resnet34',
                 pretrained=False):
        # super(PSPNet, self).__init__()
        super(UNetResNet34, self).__init__()
        self.num_classes = num_classes
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            # nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, data_dict):
        x = data_dict['image']
        h, w = x.shape[2], x.shape[3]
        # padding
        min_size = 16
        pad_h = int((h + min_size - 1) / min_size) * min_size - h
        pad_w = int((w + min_size - 1) / min_size) * min_size - w
        if pad_h > 0 or pad_w > 0:
            # Pad 0 here. Not sure whether has a large effect
            x = F.pad(x, [0, pad_w, 0, pad_h])
        # assert h % 16 == 0 and w % 16 == 0

        preds = dict()

        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)


        # crop
        if pad_h > 0 or pad_w > 0:
            p = p[:, :, 0:h, 0:w]

        seg_logit = self.final(p)

        preds['seg_logit'] = seg_logit

        preds['feature'] = p

        return preds


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

if __name__ == '__main__':
    b, c, h, w = 2, 20, 120, 160
    image = torch.randn(b, 3, h, w).cuda()
    net = PSPNet(c, pretrained=False)
    net.cuda()
    preds = net({'image': image})
    for k, v in preds.items():
        print(k, v.shape)
