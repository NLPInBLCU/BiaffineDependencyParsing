# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     load_model
   Description :
   Author :       Liangs
   date：          2019/8/5
-------------------------------------------------
   Change Activity:
                   2019/8/5:
-------------------------------------------------
"""
from models.hdlstm_biaffine import HDLSTMBiaffine
from common.model_utils import LSTMModelWrapper


def load_model(args, vocab=None, pretrain=None):
    if args.mode == 'train':
        if args.model == 'transformer':
            pass
        elif args.model == 'hdlstm':
            model = HDLSTMBiaffine(args, vocab, emb_matrix=pretrain.emb)
            args.model_class = HDLSTMBiaffine
            args.pretrain = pretrain
            args.vocab = vocab
            model_wrapper = LSTMModelWrapper(args, model)
        else:
            raise ValueError('illegal model config')
    elif args.mode == 'inference' and args.saved_model_file:
        if args.model == 'transformer':
            pass
        elif args.model == 'hdlstm':
            model_wrapper = LSTMModelWrapper(args)
        else:
            raise ValueError('illegal model config')
    else:
        raise ValueError('bad model config')

    return model_wrapper
