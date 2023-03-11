import torch
import torch.nn as nn
import torch.nn.functional as F

def CrossEntropy_Loss():
    return torch.nn.CrossEntropyLoss()

class OhemCrossEntropy2dTensor(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

class CriterionDSN(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255,reduce=True):
        super(CriterionDSN, self).__init__()

        self.ignore_index = ignore_index
        self.reduce = reduce
    def forward(self, preds, target):
        scale_pred = preds[0]
        loss1 = super(CriterionDSN, self).forward(scale_pred, target)
        scale_pred = preds[1]
        loss2 = super(CriterionDSN, self).forward(scale_pred, target)

        return loss1 + loss2 * 0.4


class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the models.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh=thresh, min_kept=min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2 * 0.4  

#focal loss(预防正负样本不平衡)
def focal_loss(output, target,alpha=0.5, gamma=2):
    logpt  = -nn.CrossEntropyLoss(reduction='none')(output, target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def target_trans(target,num_classes):
    # target.shape为（n,h,w,class+1）
    for i in range(target.shape[0]):
        label = target[i].long()
        h,w = label.shape
        label[label >= num_classes] = num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        # a = 
        if torch.cuda.is_available():
            seg_labels  = torch.eye(num_classes + 1).to(torch.device('cuda:0'))[label.reshape([-1])]
        else:
            seg_labels  = torch.eye(num_classes + 1)[label.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(h), int(w),num_classes + 1))
        if i ==0:
            final_torch =  torch.unsqueeze(seg_labels,dim=0)
        else:
            temp_torch2 = torch.unsqueeze(seg_labels,dim=0)
            final_torch = torch.cat((final_torch,temp_torch2),dim=0)
    return final_torch

#dice_loss
def Dice_loss(inputs, target, beta=1, smooth = 1e-5,num_classes=2):
    #转换traget向量
    temp_torch = target_trans(target,num_classes)
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = temp_torch.size()

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = temp_torch.view(n, -1, ct)
    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss