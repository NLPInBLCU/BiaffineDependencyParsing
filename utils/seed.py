# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/9
import numpy as np
import torch
import random


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
    pass
