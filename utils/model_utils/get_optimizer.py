# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/7
import re
import math
import torch
# from model_utils.optimization import *
import pytorch_transformers.optimization as huggingfaceOptim  # 避免和torch.optim重名
from utils.information import debug_print


def get_optimizer_old(name, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters)  # use default lr
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def _get_bertology_optimizer_grouped_parameters(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters


def _get_bertology_different_lr_grouped_parameters(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    params_bert_no_decay = []
    params_bert_decay = []
    params_other_no_decay = []
    params_other_decay = []
    for n, p in model.named_parameters():
        if 'encoder.bertology.' in n:
            # 是BERTology的参数
            if any(nd in n for nd in no_decay):
                # no decay
                params_bert_no_decay.append(p)
            else:
                params_bert_decay.append(p)
        else:
            # 不是bertology的参数
            if any(nd in n for nd in no_decay):
                # no decay
                params_other_no_decay.append(p)
            else:
                params_other_decay.append(p)
    optimizer_grouped_parameters = [
        {
            'params': params_bert_decay,
            'weight_decay': args.weight_decay,
            'lr': args.bertology_lr,
        },
        {
            'params': params_bert_no_decay,
            'weight_decay': 0.0,
            'lr': args.bertology_lr,
        },
        {
            'params': params_other_decay,
            'weight_decay': args.weight_decay,
            'lr': args.other_lr,
        },
        {
            'params': params_other_no_decay,
            'weight_decay': 0.0,
            'lr': args.other_lr,
        },
    ]
    return optimizer_grouped_parameters


def get_optimizer(args, model):
    args.warmup_steps = math.ceil(args.warmup_prop * args.max_train_steps)
    if args.optimizer == 'adamw-bertology':
        if args.different_lr:
            optimizer_grouped_parameters = _get_bertology_different_lr_grouped_parameters(args, model)
        else:
            optimizer_grouped_parameters = _get_bertology_optimizer_grouped_parameters(args, model)
        optimizer = huggingfaceOptim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                                           betas=(args.beta1, args.beta2))
        scheduler = huggingfaceOptim.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                                          t_total=args.max_train_steps)
        debug_print('\n - Use Huggingface\'s AdamW Optimizer')
    elif args.optimizer == 'adamw-torch':
        try:
            from torch.optim import AdamW
        except ImportError as e:
            debug_print(f'torch version: {torch.__version__}')
            raise e
        if args.different_lr:
            optimizer_grouped_parameters = _get_bertology_different_lr_grouped_parameters(args, model)
        else:
            optimizer_grouped_parameters = _get_bertology_optimizer_grouped_parameters(args, model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                          betas=(args.beta1, args.beta2))
        scheduler = huggingfaceOptim.WarmupLinearSchedule(optimizer,
                                                          warmup_steps=args.warmup_steps,
                                                          t_total=args.max_train_steps)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps,
                                     weight_decay=args.weight_decay)
        scheduler = None
    elif args.rnn_optimizer == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters())  # use default lr
        scheduler = None
    else:
        raise Exception("Unsupported optimizer: {}".format(args.optimizer))
    return optimizer, scheduler


if __name__ == '__main__':
    pass
