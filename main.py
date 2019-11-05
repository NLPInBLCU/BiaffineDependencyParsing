# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main.py
   Description :
   Author :       Liangs
   date：          2019/7/28
-------------------------------------------------
   Change Activity:
                   2019/10/29:
-------------------------------------------------
"""
import os
import random
import torch
import numpy as np
import pathlib
import shutil
from datetime import datetime
from utils.arguments import parse_args
from models.biaffine_trainer import BERTBiaffineTrainer
from models.biaffine_model import BiaffineDependencyModel
from utils.input_utils.bert.bert_input_utils import load_input
from utils.input_utils.graph_vocab import GraphVocab
from utils.seed import set_seed
from utils.timer import Timer
from utils.logger import init_logger, get_logger


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
    # 创建输出文件夹，保存运行结果，配置文件，模型参数
    make_output_dir(args)
    # 添加多卡运行下的配置参数
    # BERT必须在多卡下运行，单卡非常慢
    config_for_multi_gpu(args)
    # set_seed 必须在设置n_gpu之后
    set_seed(args)

    with Timer('load input'):
        train_data_loader, train_conllu, dev_data_loader, dev_conllu, _, _ = load_input(args,
                                                                                        train=True,
                                                                                        dev=True,
                                                                                        test=False)
    print(f'train batch size: {args.train_batch_size}')
    print(f'train data batch num: {len(train_data_loader)}')
    # 每个epoch做两次dev：
    args.eval_interval = len(train_data_loader) // 2
    print(f'eval interval: {args.eval_interval}')
    # 注意该参数影响学习率warm up
    args.max_train_steps = len(train_data_loader) * args.max_train_epochs
    print(f'max steps: {args.max_train_steps}')
    # 如果6个epoch之后仍然不能提升，就停止
    if args.early_stop:
        args.early_stop_steps = len(train_data_loader) * args.early_stop_epochs
        print(f'early stop steps: {args.early_stop_steps}\n')
    else:
        print(f'do not use early stop, training will last {args.max_train_epochs} epochs')

    with Timer('load trainer'):
        trainer = load_trainer(args)
    with Timer('Train'):
        trainer.train(train_data_loader, dev_data_loader, dev_conllu)


def config_for_multi_gpu(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)


def make_output_dir(args):
    output_dir = pathlib.Path(args.output_dir)
    assert output_dir.is_dir()
    time_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    output_dir = output_dir / (pathlib.Path(args.config_file).stem + time_str)
    if output_dir.exists():
        raise RuntimeError(f'{output_dir} exists! (maybe file or dir)')
    else:
        output_dir.mkdir()
        # 复制对应的配置文件到保存的文件夹下，保持配置和输出结果的一致
        shutil.copyfile(args.config_file, str(output_dir / pathlib.Path(args.config_file).name))
        (output_dir / 'saved_models').mkdir()
        args.output_dir = str(output_dir)
        args.dev_output_path = str(output_dir / 'dev_output_conllu.txt')
        args.dev_result_path = str(output_dir / 'dev_best_metrics.txt')
        args.test_output_path = str(output_dir / 'test_output_conllu.txt')
        args.test_result_path = str(output_dir / 'test_metrics.txt')
        args.save_model_dir = str(output_dir / 'saved_models')
        args.summary_dir = str(output_dir / 'summary')
        init_logger(args.log_name, str(output_dir / 'parser.log'))


if __name__ == '__main__':
    main()
