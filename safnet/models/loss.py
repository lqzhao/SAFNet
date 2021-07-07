import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb


class SegLoss(nn.Module):
    """Segmentation loss"""

    def __init__(self, weight=None, ignore_index=-100):
        super(SegLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.chamfer = ChamferLoss()

    def forward(self, preds, data_batch):
        loss_dict = dict()
        # pdb.set_trace()
        logits = preds["seg_logit"]
        labels = data_batch["seg_label"]
        seg_loss = F.cross_entropy(logits, labels,
                                   weight=self.weight,
                                   ignore_index=self.ignore_index)
        loss_dict['seg_loss_3d'] = seg_loss

        if ("logit_2d" in preds.keys()) and ("label_2d" in data_batch.keys()):
            logits = preds["logit_2d"]
            b,c,h,w = logits.shape
            logits = logits.permute(0,1,3,2)
            labels = data_batch["label_2d"]
            labels = labels.reshape([-1] + list(labels.shape[2:]))
            labels = labels.long()
            # b,_,h,w = labels.shape
            # labels = labels.reshape(-1, 1, h,w).squeeze().long()
            # pdb.set_trace()
            seg_loss = F.cross_entropy(logits, labels,
                                       weight=self.weight,
                                       ignore_index=self.ignore_index)
            loss_dict['seg_loss_2d'] = seg_loss
        if "logit_2d_chunks" in preds.keys():
            logits = preds["logit_2d_chunks"]
            labels = data_batch["seg_label"]
            seg_loss = F.cross_entropy(logits, labels,
                                       weight=self.weight,
                                       ignore_index=self.ignore_index,reduction='none')
            # pdb.set_trace()
            loss_dict['real_seg_loss_2d_3d'] = seg_loss.mean()
            dist_mask = data_batch["knn_dist"]
            seg_loss = seg_loss * dist_mask.float()
            loss_dict['seg_loss_2d_3d'] = seg_loss.mean()

        if "logit_point_branch" in preds.keys():
            logits = preds["logit_point_branch"]
            labels = data_batch["seg_label"]
            seg_loss = F.cross_entropy(logits, labels,
                                       weight=self.weight,
                                       ignore_index=self.ignore_index)
            loss_dict['seg_loss_point_branch'] = seg_loss
        # self.chamfer(recon1, points_gt)
        # num_points = preds['real_recon'].shape[2]
        # real_recon = preds['real_recon'].view(b,3,-1).permute(0,2,1)
        # image_recon = preds['image_recon'].view(b,3,-1).permute(0,2,1)
        # realneighbors = preds['realneighbors'].view(b,3,-1).permute(0,2,1)
        # imageneighbors = preds['imageneighbors'].view(b,3,-1).permute(0,2,1)
        # # pdb.set_trace()

        # recon_loss = self.chamfer(real_recon,realneighbors) + \
        #              self.chamfer(image_recon, imageneighbors)
        # loss_dict['recon_loss'] = recon_loss

        return loss_dict

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, weight=None,ignore_index=-100):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = 8192
        # self.ignore_lb = ignore_lb
        # self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, preds, data_batch):
        # N, C, H, W = logits.size()
        # loss = self.criteria(logits, labels).view(-1)
        # loss = F.cross_entropy(logits, labels,
        #                            weight=self.weight,
        #                            ignore_index=self.ignore_index)
        # loss, _ = torch.sort(loss, descending=True)
        # if loss[self.n_min] > self.thresh:
        #     loss = loss[loss>self.thresh]
        # else:
        #     loss = loss[:self.n_min]
        # return torch.mean(loss)

        loss_dict = dict()
        logits = preds["seg_logit"]
        labels = data_batch["seg_label"]
        loss = F.cross_entropy(logits, labels,
                                   weight=self.weight,
                                   ignore_index=self.ignore_index,reduction='none').view(-1)
        loss2 = F.cross_entropy(logits, labels,
                                   weight=self.weight,
                                   ignore_index=self.ignore_index)

        loss_dict['normal_seg_loss_3d'] = loss2

        loss, _ = torch.sort(loss, descending=True)
        # print(loss)
        # pdb.set_trace()
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        loss_dict['seg_loss_3d'] = torch.mean(loss)

        if "logit_2d" in preds.keys():
            logits = preds["logit_2d"]
            b,c,h,w = logits.shape
            logits = logits.permute(0,1,3,2)
            labels = data_batch["label_2d"]
            b,_,h,w = labels.shape
            labels = labels.reshape(-1, 1, h,w).squeeze().long()
            # pdb.set_trace()
            loss = F.cross_entropy(logits, labels,
                                       weight=self.weight,
                                       ignore_index=self.ignore_index,reduction='none').view(-1)
            loss, _ = torch.sort(loss, descending=True)
            if loss[self.n_min] > self.thresh:
                loss = loss[loss > self.thresh]
            else:
                loss = loss[:self.n_min]
            loss_dict['seg_loss_2d'] = torch.mean(loss)

        if "logit_2d_chunks" in preds.keys():
            logits = preds["logit_2d_chunks"]
            labels = data_batch["seg_label"]
            loss = F.cross_entropy(logits, labels,
                                       weight=self.weight,
                                       ignore_index=self.ignore_index,reduction='none').view(-1)
            loss, _ = torch.sort(loss, descending=True)
            if loss[self.n_min] > self.thresh:
                loss = loss[loss > self.thresh]
            else:
                loss = loss[:self.n_min]
            loss_dict['seg_loss_2d_3d'] = torch.mean(loss)
            # pdb.set_trace()
        if "logit_point_branch" in preds.keys():
            logits = preds["logit_point_branch"]
            labels = data_batch["seg_label"]
            seg_loss = F.cross_entropy(logits, labels,
                                       weight=self.weight,
                                       ignore_index=self.ignore_index)
            loss_dict['seg_loss_point_branch'] = seg_loss
        return loss_dict

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

