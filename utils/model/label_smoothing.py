# -*- coding: utf-8 -*-
# Created by li huayong on 2019/11/6
import torch
import torch.nn as nn
import torch.nn.functional as F


def label_smoothed_kl_div_loss(logits, target, class_num, smothing=0.0, reduction='sum'):
    """
        函数版本的LabelSmoothedKLDivLoss,两者结果完全一样
    :param logits:
    :param target:
    :param class_num:
    :param smothing:
    :param reduction
    :return:
    """

    def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method

        """
        assert 0 <= smoothing < 1
        confidence = 1.0 - smoothing
        label_shape = torch.Size((true_labels.size(0), classes))
        with torch.no_grad():
            true_dist = torch.empty(size=label_shape, device=true_labels.device)
            true_dist.fill_(smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
        return true_dist

    preds = F.log_softmax(logits)
    smoothed_target = smooth_one_hot(target, class_num, smothing)
    return F.kl_div(preds, smoothed_target, reduction=reduction)


if __name__ == '__main__':
    pass
