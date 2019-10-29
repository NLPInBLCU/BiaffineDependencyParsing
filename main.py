# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main.py
   Description :
   Author :       Liangs
   date：          2019/7/28
-------------------------------------------------
   Change Activity:
                   2019/7/28:
-------------------------------------------------
"""
import os
import random
import torch
import numpy as np
import pathlib
import shutil
from datetime import datetime
from common.arguments import parse_args
from models.biaffine_trainer import BERTBiaffineTrainer
from models.biaffine_model import BiaffineDependencyModel
from common.input_utils.bert.bert_input_utils import load_input
from common.input_utils.graph_vocab import GraphVocab
from common.seed import set_seed
from common.timer import Timer


# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.n_gpu > 0:
#         torch.cuda.manual_seed_all(args.seed)


def load_trainer(args):
    model = BiaffineDependencyModel(args)
    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f'Parallel Train, GPU num : {args.n_gpu}')
        args.parallel_train = True
    else:
        args.parallel_train = False
    trainer = BERTBiaffineTrainer(args, model)
    return trainer


def main():
    with Timer('parse args'):
        args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    assert output_dir.is_dir()
    time_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    output_dir = output_dir / (pathlib.Path(args.config_file).stem + time_str)
    if output_dir.exists():
        raise RuntimeError(f'{output_dir} exists! (maybe file or dir)')
    else:
        output_dir.mkdir()
        shutil.copyfile(args.config_file, str(output_dir / pathlib.Path(args.config_file).name))
        (output_dir / 'saved_models').mkdir()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if args.skip_too_long_input:
        print(f'skip_too_long_input is True, max_seq_len is {args.max_seq_len}')

    if args.encoder_type == 'bert':
        if not os.path.isdir(args.bert_path):
            raise ValueError(f'{args.bert_path} is not a dir or not exist !')

    with Timer('load input'):
        train_data_loader, train_conllu, dev_data_loader, dev_conllu, _, _ = load_input(args,
                                                                                        train=True, dev=True,
                                                                                        test=False)
    print(f'train batch size: {args.train_batch_size}')
    print(f'train data batch num: {len(train_data_loader)}')
    # 每个epoch做两次dev：
    args.eval_interval = len(train_data_loader) // 2
    print(f'eval interval: {args.eval_interval}')
    # 最多过100个epoch, 注意该参数影响学习率warm up
    args.max_steps = len(train_data_loader) * 100
    print(f'max steps: {args.max_steps}')
    # 如果6个epoch之后仍然不能提升，就停止
    args.early_stop_steps = len(train_data_loader) * 6
    print(f'early stop steps: {args.early_stop_steps}')

    print()
    set_seed(args)
    with Timer('load trainer'):
        trainer = load_trainer(args)
    with Timer('Train'):
        trainer.train(train_data_loader, dev_data_loader, dev_conllu)


if __name__ == '__main__':
    main()
