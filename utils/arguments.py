# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/8
# import configargparse as argparse
import os

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
    args = parser.parse_args()
    yaml_config = yaml.load(open(args.config_file, encoding='utf-8'))
    args_dict = {}
    for sub_dict in yaml_config.values():
        if sub_dict:
            for k, v in sub_dict.items():
                if k in args_dict.keys():
                    raise ValueError(f'Duplicate parameter : {k}')
                args_dict[k] = v
    args_dict['config_file'] = args.config_file
    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args_dict['device'] = device
    args = ArgsClass(args_dict)

    if args.skip_too_long_input:
        print(f'skip_too_long_input is True, max_seq_len is {args.max_seq_len}')

    return args


if __name__ == '__main__':
    from pprint import pprint

    pprint(vars(parse_args()))
