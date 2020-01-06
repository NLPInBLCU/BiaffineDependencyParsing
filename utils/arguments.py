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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', required=True)
    parser.add_argument('--run', choices=['train', 'dev', 'inference'], default='train')
    # 以下参数仅在dev或者inference时需要：
    parser.add_argument('--model_path', default=None, help='预先训练好的模型路径（文件夹）')
    parser.add_argument('--input', default=None, help='输入的CONLL-U文件，用来dev或者inference')
    parser.add_argument('--output', default=None, help='dev或者inference的输出文件')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='')
    args = parser.parse_args()
    if args.run in ['dev', 'inference']:
        assert args.model_path and args.input and args.output
        assert pathlib.Path(args.model_path).is_dir()
        assert pathlib.Path(args.input).is_file()
    yaml_config = yaml.load(open(args.config_file, encoding='utf-8'))
    args_dict = {}
    for sub_dict in yaml_config.values():
        if sub_dict:
            for k, v in sub_dict.items():
                if k in args_dict.keys():
                    raise ValueError(f'Duplicate parameter : {k}')
                args_dict[k] = v
    args_dict['config_file'] = args.config_file
    args_dict['run_mode'] = args.run
    args_dict['cuda'] = not args.use_cpu
    args_dict['cpu'] = args.use_cpu
    if args.run in ['dev', 'inference']:
        # 覆盖模型路径
        args_dict['saved_model_path'] = args.model_path
        args_dict['input_conllu_path'] = args.input
        args_dict['output_conllu_path'] = args.output
    args = ArgsClass(args_dict)

    if args.skip_too_long_input:
        print(f'skip_too_long_input is True, max_seq_len is {args.max_seq_len}')

    return args


if __name__ == '__main__':
    from pprint import pprint

    pprint(vars(parse_args()))
