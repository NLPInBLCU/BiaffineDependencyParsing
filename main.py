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
import os
import torch
import pathlib
import shutil
from datetime import datetime
import yaml

from trainers.bertology_trainer import BERTologyBaseTrainer
from models.biaffine_model import BiaffineDependencyModel
from utils.arguments import parse_args
from utils.data.bertology_loader import load_bertology_input
from PyToolkit.PyToolkit import Timer, init_logger, get_logger
from PyToolkit.PyToolkit.seed import set_seed

logger = init_logger('parser', only_console=True)


def load_trainer(configs):
    if configs.command == 'train':
        # 默认train模式下是基于原始BERT预训练模型的参数开始的
        # 实际保持initialize_from_bertology=True也没有影响（既可以从BERT模型初始化，也可以断点恢复训练）
        model = BiaffineDependencyModel.from_pretrained(configs, initialize_from_bertology=True)
    else:
        # dev或者inference模式不是从原始BERT模型初始化参数
        model = BiaffineDependencyModel.from_pretrained(configs, initialize_from_bertology=False)

    model.to(configs.device)

    # multi-gpu training (should be after apex fp16 initialization)
    if configs.n_gpu > 1 and configs.local_rank == -1:
        # 使用 DataParallel 多卡并行计算
        # 值得注意的是，模型和数据都需要先 load 进 GPU 中，
        # DataParallel 的 module 才能对其进行处理，否则会报错
        model = torch.nn.DataParallel(model)
        logger.info(f'Parallel Running, GPU num : {configs.n_gpu}')
    # Distributed training (should be after apex fp16 initialization)
    if configs.local_rank != -1:
        # 使用 DistributedDataParallel 包装模型，
        # 它能帮助我们为不同 GPU 上求得的梯度进行 all reduce（即汇总不同 GPU 计算所得的梯度，并同步计算结果）
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[configs.local_rank], output_device=configs.local_rank,
            find_unused_parameters=True,
        )

    if configs.encoder_type == 'bertology':
        trainer = BERTologyBaseTrainer(configs, model)
    else:
        raise ValueError('Encoder Type not supported temporarily')
    return trainer


def setup_for_multi_gpu(configs):
    # Setup CUDA, GPU & distributed training
    if configs.local_rank == -1 or configs.cpu:
        configs.device = torch.device("cuda" if torch.cuda.is_available() and not configs.cpu else "cpu")
        configs.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(configs.local_rank)
        configs.device = torch.device("cuda", configs.local_rank)
        # 使用 init_process_group 设置GPU 之间通信使用的后端和端口
        torch.distributed.init_process_group(backend="nccl")
        # # 分布式训练时，n_gpu设置为1
        configs.n_gpu = 1
    configs.train_batch_size = configs.per_gpu_train_batch_size * max(1, configs.n_gpu)
    configs.eval_batch_size = configs.per_gpu_eval_batch_size * max(1, configs.n_gpu)


def setup_output_dir(configs):
    global logger
    assert configs.command == 'train', '仅在train模式下保存各种输出文件'
    if configs.no_output:
        logger = init_logger(configs.log_name, only_console=True)
        return
    output_dir = pathlib.Path(configs.output_dir)
    assert output_dir.is_dir()
    time_str = datetime.now().strftime('_%Y-%m-%d-%H-%M-%S')
    output_dir = output_dir / (pathlib.Path(configs.config_file).stem + time_str)
    if output_dir.exists():
        raise RuntimeError(f'{output_dir} exists! (maybe file or dir)')
    else:
        output_dir.mkdir()
        # 复制对应的配置文件到保存的文件夹下，保持配置和输出结果的一致
        shutil.copyfile(configs.config_file, str(output_dir / pathlib.Path(configs.config_file).name))
        # 复制graphVocab到输出文件下：
        shutil.copyfile(configs.graph_vocab_file, str(output_dir / pathlib.Path(configs.graph_vocab_file).name))
        (output_dir / 'model').mkdir()
        configs.output_dir = str(output_dir)
        configs.dev_output_path = str(output_dir / 'dev_output_conllu.txt')
        configs.dev_result_path = str(output_dir / 'dev_best_metrics.txt')
        configs.test_output_path = str(output_dir / 'test_output_conllu.txt')
        configs.test_result_path = str(output_dir / 'test_metrics.txt')
        configs.output_model_dir = str(output_dir / 'model')
        configs.summary_dir = str(output_dir / 'summary')
        logger = init_logger(configs.log_name, str(output_dir / 'parser.log'))


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


def train(configs) -> None:
    """
    训练模式

    Args:
        configs: 配置参数

    Returns:
        None
    """
    if configs.command != 'train':
        raise RuntimeError('Not in train mode')

    with Timer('Load data set'):
        # 目前仅仅支持BERTology形式的输入
        train_data_loader, _, dev_data_loader, dev_conllu = load_bertology_input(configs)

    logger.info(f'train batch size: {configs.train_batch_size}')
    logger.info(f'train data batch num: {len(train_data_loader)}')
    # dev的间隔步数：
    configs.eval_interval = len(train_data_loader) * configs.eval_epoch
    logger.info(f'eval interval: {configs.eval_interval}')
    # 注意该参数影响学习率warm up
    configs.max_train_steps = len(train_data_loader) * configs.max_train_epochs
    logger.info(f'max steps: {configs.max_train_steps}')
    # 如果6个epoch之后仍然不能提升，就停止
    if configs.early_stop:
        logger.info(f'early stop steps: {configs.early_stop_epochs}\n')
    else:
        logger.info(f'do not use early stop, training will last {configs.max_train_epochs} epochs')
    with Timer('Load trainer'):
        trainer = load_trainer(configs)

    save_config_to_yaml(configs)

    with Timer('Train'):
        trainer.train(train_data_loader, dev_data_loader, dev_conllu)
    logger.info('Train Complete!')

    if configs.test_after_train and configs.local_rank in [-1, 0]:
        if configs.no_output:
            raise RuntimeError('no_output为True时无法训练后立刻测试')
        # 最优模型的保存位置
        configs.saved_model_path = configs.output_model_dir
        # 测试gold文件
        configs.input_conllu_path = os.path.join(configs.data_dir, configs.test_file)
        configs.output_conllu_path = configs.test_output_path
        configs.command = 'test_after_train'
        dev(configs)


def dev(configs):
    """
    验证模式，gold input file: configs.input_conllu_path; dev output file: configs.output_conllu_path
    Args:
        configs:

    Returns:

    """
    if configs.command not in ['dev', 'test_after_train']:
        raise RuntimeError('Not in dev mode')
    dev_data_loader, dev_conllu = load_bertology_input(configs)
    with Timer('Load trainer'):
        trainer = load_trainer(configs)
    with Timer('dev'):
        dev_UAS, dev_LAS = trainer.dev(dev_data_loader, dev_conllu,
                                       input_conllu_path=configs.input_conllu_path,
                                       output_conllu_path=configs.output_conllu_path)
    print(f'DEV output file saved in {configs.output_conllu_path}')
    print(f'DEV metrics:\nUAS:{dev_UAS}\nLAS:{dev_LAS}')


def inference(configs):
    if configs.command != 'inference':
        raise RuntimeError('Not in inference mode')
    inference_data_loader, inference_conllu = load_bertology_input(configs)
    with Timer('load trainer'):
        trainer = load_trainer(configs)
    with Timer('inference'):
        trainer.inference(inference_data_loader, inference_conllu, output_conllu_path=configs.output_conllu_path)
    print(f'INFERENCE output file saved in {configs.output_conllu_path}')


def main():
    with Timer('Parse args'):
        # 加载参数设置
        configs = parse_args()
    # 添加多卡运行下的配置参数, Setup CUDA, GPU & distributed training
    setup_for_multi_gpu(configs)
    # set_seed 必须在设置n_gpu之后
    set_seed(configs)
    # 训练模式下需要创建输出文件夹，以用来保存运行结果，配置文件，模型参数等
    if configs.command == 'train' and configs.local_rank in [-1, 0]:
        setup_output_dir(configs)

    if configs.command == 'train':
        train(configs)
    elif configs.command == 'dev':
        dev(configs)
    # 支持训练完成之后立刻在test上测试结果
    elif configs.command == 'inference':
        inference(configs)


if __name__ == '__main__':
    # todo:重写conllu文件的加载、写入、指标计算
    main()
