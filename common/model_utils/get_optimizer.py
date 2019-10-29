# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/7
import torch
# from model_utils.optimization import *
from pytorch_transformers.optimization import *


# def get_optimizer_bert_old(args, model):
#     if args.optim == 'adam':
#         print("Adam Training......")
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     elif args.optim == 'optim':
#         print("SGD Training.......")
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
#                                     momentum=args.momentum_value)
#     elif args.optim == 'adadelta':
#         print("Adadelta Training.......")
#         optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     elif args.optim == 'bertadam':
#         print("BERT Adam Training...")
#         param_optimizer = list(model.named_parameters())
#         no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
#         # print([n for n, p in param_optimizer if any(nd in n for nd in no_decay)])
#         optimizer = BertAdam(optimizer_grouped_parameters,
#                              lr=args.lr,
#                              warmup=args.warmup_proportion,
#                              t_total=args.num_train_optimization_steps)
#     else:
#         raise ValueError('illegal optim setting')
#     return optimizer
from common.information import debug_print


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


def get_optimizer(args, model):
    if args.encoder_type in ['bert', 'xlnet', 'xlm', 'roberta'] or args.optimizer == 'adamw':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.max_steps)
        debug_print('\n - Use AdamW Optimizer')
    elif args.encoder_type in ['lstm', 'gru']:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps,
                                         weight_decay=args.weight_decay)
        elif args.rnn_optimizer == 'adamax':
            optimizer = torch.optim.Adamax(model.parameters())  # use default lr
        else:
            raise Exception("Unsupported optimizer: {}".format(args.rnn_optimizer))
        scheduler = None
    else:
        raise Exception('bad encoder_type')
    return optimizer, scheduler


if __name__ == '__main__':
    pass
