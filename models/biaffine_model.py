# -*- coding: utf-8 -*-
# Created by li huayong on 2019/10/7
import os
import pathlib

import torch
import torch.nn as nn

from utils.input_utils.bertology.bert_input_utils import load_bert_tokenizer, load_and_cache_examples, get_data_loader
from utils.input_utils.graph_vocab import GraphVocab
from modules.bertology_encoder import BERTologyEncoder
from modules.biaffine import DeepBiaffineScorer, DirectBiaffineScorer
from models.base_model import BaseModel


class BiaffineDependencyModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.graph_vocab = GraphVocab(args.graph_vocab_file)
        if args.encoder_type == 'bertology':
            self.encoder = BERTologyEncoder(no_cuda=not args.cuda,
                                            bertology_path=args.saved_model_path,
                                            bertology_word_select_mode=args.bertology_word_select,
                                            bertology_output_mode=args.bertology_output_mode,
                                            max_seq_len=args.max_seq_len,
                                            bertology_after=args.bertology_after,
                                            after_layers=args.after_layers,
                                            after_dropout=args.after_dropout)
        elif args.encoder_type in ['lstm', 'gru']:
            self.encoder = None  # Do NOT support now #todo
        elif args.encoder_type == 'transformer':
            self.encoder = None  # Do NOT support now #todo
        if args.direct_biaffine:
            self.unlabeled_biaffine = DirectBiaffineScorer(args.encoder_output_dim,
                                                           args.encoder_output_dim,
                                                           1, pairwise=True)
            self.labeled_biaffine = DirectBiaffineScorer(args.encoder_output_dim,
                                                         args.encoder_output_dim,
                                                         len(self.graph_vocab.get_labels()),
                                                         pairwise=True)
        else:
            self.unlabeled_biaffine = DeepBiaffineScorer(args.encoder_output_dim,
                                                         args.encoder_output_dim,
                                                         args.biaffine_hidden_dim,
                                                         1, pairwise=True,
                                                         dropout=args.biaffine_dropout)
            self.labeled_biaffine = DeepBiaffineScorer(args.encoder_output_dim,
                                                       args.encoder_output_dim,
                                                       args.biaffine_hidden_dim,
                                                       len(self.graph_vocab.get_labels()),
                                                       pairwise=True,
                                                       dropout=args.biaffine_dropout)
        # self.dropout = nn.Dropout(args.dropout)
        if args.learned_loss_ratio:
            self.label_loss_ratio = nn.Parameter(torch.Tensor([0.5]))
        else:
            self.label_loss_ratio = args.label_loss_ratio

    def forward(self, inputs):
        assert isinstance(inputs, dict)
        encoder_output = self.encoder(**inputs)
        unlabeled_scores = self.unlabeled_biaffine(encoder_output, encoder_output).squeeze(3)
        labeled_scores = self.labeled_biaffine(encoder_output, encoder_output)
        return unlabeled_scores, labeled_scores

    # def save_pretrained(self, save_directory, weight_file_name="pytorch_model.bin"):
    #     """ Save a model and its configuration file to a directory, so that it
    #     """
    #     assert os.path.isdir(save_directory), \
    #         "Saving path should be a directory where the model and configuration can be saved"
    #
    #     # Only save the model it-self if we are using distributed training
    #     model_to_save = self.module if hasattr(self, 'module') else self
    #
    #     # Save configuration file
    #     if self.args.encoder in ['bertology', 'xlnet', 'xlm', 'roberta']:
    #         model_to_save.encoder.bertology.bertology_config.save_pretrained(save_directory)
    #
    #     # If we save using the predefined names, we can load using `from_pretrained`
    #     output_model_file = os.path.join(save_directory, weight_file_name)
    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     print("Model weights saved in {}".format(output_model_file))
    #
    # @classmethod
    # def from_pretrained(cls, args, saved_model_path=None, weight_file_name="pytorch_model.bin",
    #                     initialize_from_bertology=False):
    #     """
    #         注意，这里不支持训练BERT的LM和Next Sentence任务
    #     :param args:
    #     :param saved_model_path:
    #     :param weight_file_name:
    #     :param initialize_from_bertology:
    #     :return:
    #     """
    #     import re
    #     from collections import OrderedDict
    #
    #     if saved_model_path:
    #         args.saved_model_path = saved_model_path
    #     model = cls(args)
    #     resolved_archive_file = pathlib.Path(args.saved_model_path) / weight_file_name
    #     assert resolved_archive_file.exists()
    #     state_dict = torch.load(str(resolved_archive_file), map_location='cpu')
    #     # Convert old format to new format if needed from a PyTorch state_dict
    #     old_keys = []
    #     new_keys = []
    #     for key in state_dict.keys():
    #         new_key = None
    #         if 'gamma' in key:
    #             new_key = key.replace('gamma', 'weight')
    #         if 'beta' in key:
    #             new_key = key.replace('beta', 'bias')
    #         if new_key:
    #             old_keys.append(key)
    #             new_keys.append(new_key)
    #     for old_key, new_key in zip(old_keys, new_keys):
    #         state_dict[new_key] = state_dict.pop(old_key)
    #     # 如果模型从BERTology预训练模型加载，则需要修改参数的名称以匹配现有的模型架构
    #     if initialize_from_bertology and args.encoder_type in ['bertology', 'xlnet', 'xlm', 'roberta']:
    #         rename_state_dict = OrderedDict()
    #         for key in state_dict.keys():
    #             # 这里 ^bertology 表示以bert开头的模型参数，目前仅仅支持BERT类型的模型，XLNET等需要补充
    #             rename_key = re.sub('^bertology.', 'encoder.bertology.', key)
    #             rename_state_dict[rename_key] = state_dict[key]
    #         state_dict = rename_state_dict
    #     missing_keys = []
    #     unexpected_keys = []
    #     error_msgs = []
    #     # copy state_dict so _load_from_state_dict can modify it
    #     metadata = getattr(state_dict, '_metadata', None)
    #     state_dict = state_dict.copy()
    #     if metadata is not None:
    #         state_dict._metadata = metadata
    #
    #     def load(module, prefix=''):
    #         local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    #         module._load_from_state_dict(
    #             state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
    #         for name, child in module._modules.items():
    #             if child is not None:
    #                 load(child, prefix + name + '.')
    #
    #     load(model)
    #
    #     if len(missing_keys) > 0:
    #         print("Weights of {} not initialized from pretrained model: {}".format(
    #             model.__class__.__name__, missing_keys))
    #     if len(unexpected_keys) > 0:
    #         print("Weights from pretrained model not used in {}: {}".format(
    #             model.__class__.__name__, unexpected_keys))
    #     if len(error_msgs) > 0:
    #         raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    #             model.__class__.__name__, "\n\t".join(error_msgs)))
    #
    #     model.eval()
    #
    #     return model


if __name__ == '__main__':
    class Args():
        def __init__(self):
            self.bert_path = '/home/liangs/disk/data/bertology-base-chinese'
            self.data_dir = '../dataset'
            self.train_file = 'test.conllu'
            self.max_seq_length = 10
            self.encoder_type = 'bertology'
            self.root_representation = 'unused'
            self.graph_vocab_file = '../dataset/graph_vocab.txt'
            self.cuda = False
            self.bert_chinese_word_select = 's+e'
            self.bert_output_mode = 'last_four_sum'

            # for Biaffine
            self.biaffine_hidden_dim = 300
            self.biaffine_dropout = 0.1

            # for loss:
            self.learned_loss_ratio = True,
            self.label_loss_ratio = 0.5


    args = Args()

    if args.encoder_type == 'bertology':
        args.encoder_output_dim = 768

    tokenizer = load_bert_tokenizer('/home/liangs/disk/data/bertology-base-chinese', 'bertology')
    vocab = GraphVocab('../dataset/graph_vocab.txt')
    dataset, CoNLLU_file = load_and_cache_examples(args, vocab, tokenizer)
    data_loader = get_data_loader(dataset, batch_size=2, evaluation=True)
    # bertology = BERTTypeEncoder(no_cuda=True, bert_path=args.bert_path)
    model = BiaffineDependencyModel(args)
    print(model)
    for batch in data_loader:
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2] if args.encoder_type in ['bertology', 'xlnet'] else None,
            'start_pos': batch[3],
            'end_pos': batch[4],
        }
        # print(inputs)
        # print(inputs['start_pos'])
        output = model(inputs)
        print(output)
