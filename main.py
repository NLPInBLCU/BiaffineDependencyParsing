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
import copy
import logging
import os

import torch
import pathlib
import shutil
from datetime import datetime

import yaml

from utils.arguments import parse_args
from trainers.bertology_trainer import BERTologyBaseTrainer
from models.biaffine_model import BiaffineDependencyModel
from utils.input_utils.bertology.input_utils import load_bertology_input
from utils.seed import set_seed
from utils.timer import Timer
from utils.logger import init_logger, get_logger


def load_trainer(args):
    logger = get_logger(args.log_name)

    if args.run_mode == 'train':
        # 默认train模式下是基于原始BERT预训练模型的参数开始的
        # 实际保持initialize_from_bertology=True也没有影响（既可以从BERT模型初始化，也可以断点恢复训练）
        model = BiaffineDependencyModel.from_pretrained(args, initialize_from_bertology=True)
    else:
        model = BiaffineDependencyModel.from_pretrained(args, initialize_from_bertology=False)

    model.to(args.device)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and args.local_rank == -1:
        # 使用 DataParallel 多卡并行计算
        # 值得注意的是，模型和数据都需要先 load 进 GPU 中，
        # DataParallel 的 module 才能对其进行处理，否则会报错
        model = torch.nn.DataParallel(model)
        logger.info(f'Parallel Running, GPU num : {args.n_gpu}')
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # 使用 DistributedDataParallel 包装模型，
        # 它能帮助我们为不同 GPU 上求得的梯度进行 all reduce（即汇总不同 GPU 计算所得的梯度，并同步计算结果）
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True,
        )

    if args.encoder_type == 'bertology':
        trainer = BERTologyBaseTrainer(args, model)
    else:
        raise ValueError('Encoder Type not supported temporarily')
    return trainer


def config_for_multi_gpu(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.cpu:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        # 使用 init_process_group 设置GPU 之间通信使用的后端和端口
        torch.distributed.init_process_group(backend="nccl")
        # # 分布式训练时，n_gpu设置为1
        args.n_gpu = 1
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)


def make_output_dir(args):
    assert args.run_mode == 'train', '仅在train模式下保存各种输出文件'
    if args.no_output:
        init_logger(args.log_name, only_console=True)
        return
    output_dir = pathlib.Path(args.output_dir)
    assert output_dir.is_dir()
    time_str = datetime.now().strftime('_%Y-%m-%d-%H-%M-%S')
    output_dir = output_dir / (pathlib.Path(args.config_file).stem + time_str)
    if output_dir.exists():
        raise RuntimeError(f'{output_dir} exists! (maybe file or dir)')
    else:
        output_dir.mkdir()
        # 复制对应的配置文件到保存的文件夹下，保持配置和输出结果的一致
        shutil.copyfile(args.config_file, str(output_dir / pathlib.Path(args.config_file).name))
        # 复制graphVocab到输出文件下：
        shutil.copyfile(args.graph_vocab_file, str(output_dir / pathlib.Path(args.graph_vocab_file).name))
        (output_dir / 'model').mkdir()
        args.output_dir = str(output_dir)
        args.dev_output_path = str(output_dir / 'dev_output_conllu.txt')
        args.dev_result_path = str(output_dir / 'dev_best_metrics.txt')
        args.test_output_path = str(output_dir / 'test_output_conllu.txt')
        args.test_result_path = str(output_dir / 'test_metrics.txt')
        args.output_model_dir = str(output_dir / 'model')
        args.summary_dir = str(output_dir / 'summary')
        init_logger(args.log_name, str(output_dir / 'parser.log'))


def save_config_to_yaml(_config):
    config = copy.deepcopy(_config)
    if not isinstance(config, dict):
        config = vars(config)
    del_keys = []
    for k, v in config.items():
        if type(v) not in [list, tuple, str, int, float, bool, None]:
            del_keys.append(k)
    for k in del_keys:
        del config[k]
    with open(pathlib.Path(config['output_dir']) / 'config.yaml', 'w', encoding='utf-8')as f:
        yaml.dump(config, f)


def train(args):
    logger = get_logger(args.log_name)
    assert args.run_mode == 'train'

    with Timer('load input'):
        # 目前仅仅支持BERTology形式的输入
        train_data_loader, _, dev_data_loader, dev_conllu = load_bertology_input(args)

    logger.info(f'train batch size: {args.train_batch_size}')
    logger.info(f'train data batch num: {len(train_data_loader)}')
    # 每个epoch做两次dev：
    args.eval_interval = len(train_data_loader) // 2 if len(train_data_loader) > 100 else 30
    logger.info(f'eval interval: {args.eval_interval}')
    # 注意该参数影响学习率warm up
    args.max_train_steps = len(train_data_loader) * args.max_train_epochs
    logger.info(f'max steps: {args.max_train_steps}')
    # 如果6个epoch之后仍然不能提升，就停止
    if args.early_stop:
        args.early_stop_steps = len(train_data_loader) * args.early_stop_epochs
        logger.info(f'early stop steps: {args.early_stop_steps}\n')
    else:
        logger.info(f'do not use early stop, training will last {args.max_train_epochs} epochs')
    with Timer('load trainer'):
        trainer = load_trainer(args)
    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    save_config_to_yaml(args)
    with Timer('Train'):
        trainer.train(train_data_loader, dev_data_loader, dev_conllu)
    logger.info('train DONE')
    if args.test_after_train and not args.no_output:
        args.saved_model_path = args.output_model_dir
        args.input_conllu_path = os.path.join(args.data_dir, args.test_file)
        args.output_conllu_path = 'tmp/tmp_file'
        args.run_mode = 'dev'
        dev(args)


def dev(args):
    logger = get_logger(args.log_name)
    # args = trainer.args
    assert args.run_mode == 'dev'
    dev_data_loader, dev_conllu = load_bertology_input(args)
    with Timer('load trainer'):
        trainer = load_trainer(args)
    with Timer('dev'):
        dev_UAS, dev_LAS = trainer.dev(dev_data_loader, dev_conllu,
                                       input_conllu_path=args.input_conllu_path,
                                       output_conllu_path=args.output_conllu_path)
    logger.info(f'DEV output file saved in {args.output_conllu_path}')
    logger.info(f'DEV metrics:\nUAS:{dev_UAS}\nLAS:{dev_LAS}')


def inference(args):
    logger = get_logger(args.log_name)
    # args = trainer.args
    assert args.run_mode == 'inference'
    inference_data_loader, inference_conllu = load_bertology_input(args)
    with Timer('load trainer'):
        trainer = load_trainer(args)
    with Timer('inference'):
        trainer.inference(inference_data_loader, inference_conllu, output_conllu_path=args.output_conllu_path)
    logger.info(f'INFERENCE output file saved in {args.output_conllu_path}')


def main():
    with Timer('parse args'):
        args = parse_args()
    # 添加多卡运行下的配置参数
    # Setup CUDA, GPU & distributed training
    config_for_multi_gpu(args)
    # set_seed 必须在设置n_gpu之后
    set_seed(args)
    # 创建输出文件夹，保存运行结果，配置文件，模型参数
    if args.run_mode == 'train' and args.local_rank in [-1, 0]:
        make_output_dir(args)

    if args.run_mode == 'train':
        train(args)
    elif args.run_mode == 'dev':
        dev(args)
    elif args.run_mode == 'inference':
        inference(args)


if __name__ == '__main__':
    main()
