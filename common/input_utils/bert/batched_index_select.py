# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24
# Batched index_select
import torch


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


if __name__ == '__main__':
    pass
