# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     arguments
   Description :
   Author :       Liangs
   date：          2019/7/27
-------------------------------------------------
   Change Activity:
                   2019/7/27:
-------------------------------------------------
"""
import configargparse as argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', is_config_file=True, required=True)
    # 数据:
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--train_file', type=str, default='train/text_news.train.conllu')
    parser.add_argument('--dev_file', type=str, default='dev/sdp_text_dev.conllu')
    parser.add_argument('--test_file', type=str, default='test/sdp_text_test.conllu')
    parser.add_argument('--word_cutoff', type=int, default=0)
    parser.add_argument('--label_cutoff', type=int, default=0)
    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    # 资源:
    parser.add_argument('--vectors_dir', type=str, default='embeddings', help='词向量文件夹')
    parser.add_argument('--pretrain_file', type=str, default='sem16_tencent.pkl', help='词向量文件')
    # run:
    parser.add_argument('--mode', default='train', choices=['train', 'inference'])
    parser.add_argument('--model', default='hdlstm', choices=['transformer', 'hdlstm'])

    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=500000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
    parser.add_argument('--logger_name', type=str, default='CCSD')
    parser.add_argument('--max_steps_before_stop', type=int, default=4000)

    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--dev_batch_size', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='saved_models/', help='Root dir for saving models.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    # save；
    parser.add_argument('--saved_model_file', default='checkpoints/CCSD_model.pt')
    parser.add_argument('--continue_train', type=bool, default=False)
    # 模型:
    # -------------------------------------------------------------------------------------------------->>>>>>>
    #       highway-dropout-LSTM
    parser.add_argument('--word_emb_dim', type=int, default=100)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=100)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=600)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--word_dropout', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.33)

    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    #    一些开关
    parser.add_argument('--real_highway', type=bool, default=False)
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_false',
                        help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")
    # --------------------------------------------------------------------------------------------------<<<<<<

    # -------------------------------------------------------------------------------------------------->>>>>>>
    #               Transformer

    # --------------------------------------------------------------------------------------------------<<<<<<

    # optim:
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dis_lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0, help="adam beta1,normal=0.9")
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--L2_penalty', type=float, default=3e-9, help='normal=0')
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--rel_loss_ratio', type=float, default=0.5)
    # 梯度裁剪：
    parser.add_argument('--use_grad_clip', type=bool, default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')

    args = parser.parse_args()
    # print(parser.format_values())
    return args


if __name__ == '__main__':
    args = parse_args()
    from pprint import pprint

    pprint(vars(args))
