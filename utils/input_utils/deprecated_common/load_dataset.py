# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     load_dataset
   Description :
   Author :       Liangs
   date：          2019/8/5
-------------------------------------------------
   Change Activity:
                   2019/8/5:
-------------------------------------------------
"""
from utils.input_utils.deprecated_common import DataLoader
from utils.input_utils.deprecated_common import Pretrain
from pathlib import Path


def load_data(args):
    train_file_path = str(Path(args.data_dir) / args.train_file)
    dev_file_path = str(Path(args.data_dir) / args.dev_file)
    test_file_path = str(Path(args.data_dir) / args.test_file)
    pretrain_file_path = str(Path(args.vectors_dir) / args.pretrain_file)
    pretrain_vectors = Pretrain(pretrain_file_path, args.logger_name)
    train_iter = DataLoader(train_file_path, args.batch_size, args, pretrain_vectors, evaluation=False)
    train_vocab = train_iter.vocab
    dev_iter = DataLoader(dev_file_path, args.dev_batch_size, args,
                          pretrain_vectors, vocab=train_vocab,
                          evaluation=True)
    test_iter = DataLoader(test_file_path, args.dev_batch_size, args,
                           pretrain_vectors, vocab=train_vocab,
                           evaluation=True)
    return train_iter, dev_iter, test_iter, pretrain_vectors, train_vocab


def load_test_data(args, pretrain, vocab):
    test_file_path = str(Path(args.data_dir) / args.test_file)
    test_iter = DataLoader(test_file_path, args.dev_batch_size, args, pretrain, vocab=vocab, evaluation=True)
    return test_iter
