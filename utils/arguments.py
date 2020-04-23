# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/8
# import configargparse as argparse
from pathlib import Path
import argparse
import yaml
from types import SimpleNamespace
from typing import Dict, List, Tuple


def load_configs_from_yaml(yaml_file: str) -> Dict:
    """
    从yaml配置文件中加载参数，这里会将嵌套的二级映射调整为一级映射

    Args:
        yaml_file: yaml】文件路径

    Returns:
        yaml文件中的配置字典
    """
    yaml_config = yaml.load(open(yaml_file, encoding='utf-8'), Loader=yaml.FullLoader)
    configs_dict = {}
    for sub_k, sub_v in yaml_config.items():
        # 读取嵌套的参数
        if isinstance(sub_v, dict):
            for k, v in sub_v.items():
                if k in configs_dict.keys():
                    raise ValueError(f'Duplicate parameter : {k}')
                configs_dict[k] = v
        else:
            configs_dict[sub_k] = sub_v
    return configs_dict


def parse_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser()

    # How to set `local_rank` argument?
    # Ref: https://github.com/huggingface/transformers/issues/1651
    #     The easiest way is to use the torch launch script.
    #     It will automatically set the local rank correctly.
    #     It would look something like this:
    #         `python -m torch.distributed.launch --nproc_per_node 8 run_squad.py <your arguments>`
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank, torch.distributed.launch 启动器会自动赋值")
    parser.add_argument('--cpu', action='store_true', default=False, help='不使用cuda加速，仅使用cpu')

    subparsers = parser.add_subparsers(help='子命令：{train:训练|dev:验证|infer:推理}', dest='command')
    # ------------------------训练模式参数---------------------------------------------------------------
    parser_train = subparsers.add_parser('train', help='训练模式')
    parser_train.add_argument('-c', '--config_file', required=True, help='训练时的yaml配置文件路径')
    parser_train.add_argument('--no_output', action='store_true', default=False, help='不输出文件，用于调试')
    parser_train.add_argument('--test_after_train', action='store_true', help='是否在训练完成之后立刻做一次测试')
    # ------------------------验证/推理模式参数-------------------------------------------------------------
    # 验证模式和推理模式所需要的参数完全相同（--saved_model_path,-i,-o）,因此这里使用一个parent_parser来处理重复的参数
    # 更详细的说明参考：https://stackoverflow.com/a/56595689 [Python argparse - Add argument to multiple subparsers]
    dev_infer_parent_parser = argparse.ArgumentParser(add_help=False)  # add_help必须为False
    dev_infer_parent_parser.add_argument('-m', '--saved_model_path', required=True,
                                         help='预先训练好的模型路径，其下需包含一个config.yaml(配置信息)文件和一个model(模型参数)文件夹')
    dev_infer_parent_parser.add_argument('-i', '--input_conllu_path', required=True,
                                         help='输入CONLL-U文件，dev模式下是一个gold file，infer模式下是一个空conllu file')
    dev_infer_parent_parser.add_argument('-o', '--output_conllu_path', required=True, help='dev或者infer的输出文件路径')
    dev_infer_parent_parser.add_argument('-b', '--batch_size', default=None, help='dev或者infer时刻的batch大小')
    # -----------------------再处理dev和infer各自的参数（如果有）--------------------------------------------
    parser_dev = subparsers.add_parser('dev', help='验证模式', parents=[dev_infer_parent_parser])
    parser_infer = subparsers.add_parser('infer', help='推理模式', parents=[dev_infer_parent_parser])
    # --------------------------------------------------------------------------------------------------

    configs = vars(parser.parse_args())
    configs['cuda'] = not configs['cpu']

    # 加载yaml配置参数
    if configs['command'] == 'train':
        yaml_config_file = Path(configs['config_file'])
    else:
        # dev 或者 inference 模式下配置文件config.yaml放置在saved_model_path文件夹下
        # 而模型参数则放置在 saved_model_path/model 下
        yaml_config_file = Path(configs['saved_model_path']) / 'config.yaml'
        # 获取yaml配置文件之后将 saved_model_path 调整为 saved_model_path/model，指向真正放置模型参数的文件夹
        configs['saved_model_path'] = str(Path(configs['saved_model_path']) / 'model')
        if configs['batch_size']:
            configs['eval_batch_size'] = configs['batch_size']
        del configs['batch_size']

    if not yaml_config_file.exists():
        raise RuntimeError(f'yaml config file {str(yaml_config_file)} not exist')
    yaml_configs = load_configs_from_yaml(str(yaml_config_file))

    # 合并两个参数字典
    for c, v in configs.items():
        # 将命令行读取的参数字典覆盖写入到yaml配置参数字典中
        # 在推理或者验证阶段时，yaml文件中的部分配置信息可能是训练时遗留下来的，已经过时
        # 这里覆盖yaml配置的方式可以确保覆盖这些过时的配置信息
        yaml_configs[c] = v

    # 转化为object格式
    configs = SimpleNamespace(**yaml_configs)

    # if configs.skip_too_long_input:
    #     print(f'skip_too_long_input is True, max_seq_len is {configs.max_seq_len}')

    return configs


if __name__ == '__main__':
    from pprint import pprint

    parse_args()
