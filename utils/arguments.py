# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/8
# import configargparse as argparse
import os
import pathlib

import torch
import argparse
import yaml
from configparser import ConfigParser


class ArgsClass(object):
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)


def parser_args_from_yaml(yaml_file):
    yaml_config = yaml.load(open(yaml_file, encoding='utf-8'))
    args_dict = {}
    for sub_k, sub_v in yaml_config.items():
        if isinstance(sub_v, dict):
            for k, v in sub_v.items():
                if k in args_dict.keys():
                    raise ValueError(f'Duplicate parameter : {k}')
                args_dict[k] = v
        else:
            args_dict[sub_k] = sub_v
    return args_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', required=True)
    parser.add_argument('--run', choices=['train', 'dev', 'inference'], default='train')
    """
    How to set `local_rank` argument?
    Ref:https://github.com/huggingface/transformers/issues/1651
        The easiest way is to use the torch launch script. 
        It will automatically set the local rank correctly. 
        It would look something like this:
            `python -m torch.distributed.launch --nproc_per_node 8 run_squad.py <your arguments>`
    """
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank, torch.distributed.launch 启动器会自动赋值")
    # 以下参数仅在dev或者inference时需要：
    parser.add_argument('--model_path', default=None, help='预先训练好的模型路径（文件夹）')
    parser.add_argument('--input', default=None, help='输入的CONLL-U文件，用来dev或者inference')
    parser.add_argument('--output', default=None, help='dev或者inference的输出文件')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='不使用cuda加速，仅使用cpu')
    parser.add_argument('--no_output', action='store_true', default=False, help='不输出文件，用于调试')
    args = parser.parse_args()
    if args.run in ['dev', 'inference']:
        assert args.model_path and args.input and args.output
        assert pathlib.Path(args.model_path).is_dir()
        assert pathlib.Path(args.input).is_file()
    if args.run == 'train':
        assert args.config_file is not None
        yaml_config_file = args.config_file
    else:
        assert args.model_path and args.input and args.output
        assert pathlib.Path(args.input).is_file()
        assert (pathlib.Path(args.model_path) / 'model').is_dir()
        assert (pathlib.Path(args.model_path) / 'config.yaml').is_file()
        yaml_config_file = (pathlib.Path(args.model_path) / 'config.yaml')
    args_dict = parser_args_from_yaml(yaml_config_file)
    args_dict['config_file'] = args.config_file
    args_dict['run_mode'] = args.run
    args_dict['cuda'] = not args.no_cuda
    args_dict['cpu'] = args.no_cuda
    args_dict['local_rank'] = args.local_rank
    args_dict['no_output'] = args.no_output
    if args.model_path is not None:
        args_dict['saved_model_path'] = args.model_path
    if args.run in ['dev', 'inference']:
        # 覆盖模型路径
        args_dict['input_conllu_path'] = args.input
        args_dict['output_conllu_path'] = args.output
    args = ArgsClass(args_dict)

    if args.skip_too_long_input:
        print(f'skip_too_long_input is True, max_seq_len is {args.max_seq_len}')

    return args


if __name__ == '__main__':
    from pprint import pprint

    pprint(vars(parse_args()))
