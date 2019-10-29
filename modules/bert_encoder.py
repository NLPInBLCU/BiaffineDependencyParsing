# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24
import torch
import torch.nn as nn

from pytorch_transformers import (BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer, BertModel)

from common.information import debug_print
from common.input_utils.bert.bert_input_utils import load_bert_tokenizer, get_data_loader, load_and_cache_examples
from common.input_utils.graph_vocab import GraphVocab
from modules.layer_attention import LayerAttention

BERT_MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


class BERTTypeEncoder(nn.Module):
    def __init__(
            self,
            local_rank=-1,
            no_cuda=False,
            encoder_type='bert',
            bert_path='.',
            bert_output_mode='last',
            bert_chinese_word_embedding_select_mode='s',
            layer_dropout=0.1,
    ):
        super().__init__()
        if local_rank == -1 or no_cuda:  # 单机(多卡)训练
            self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            # torch.cuda.set_device(local_rank)
            # self.device = torch.device("cuda", local_rank)
            # torch.distributed.init_process_group(backend='nccl')
            # self.n_gpu = 1
            raise RuntimeError('暂时不支持分布式训练')
        self.bert_model_type = encoder_type.lower()
        self.bert_model_path = bert_path
        self.bert_config_class, self.bert_model_class, _ = BERT_MODEL_CLASSES[self.bert_model_type]
        self.bert_config = self.bert_config_class.from_pretrained(self.bert_model_path)
        self.bert_config.output_hidden_states = True
        # self.tokenizer = self.tokenizer_class.from_pretrained(self.model_path, do_lower_case=True)
        self.bert = self.bert_model_class.from_pretrained(self.bert_model_path,
                                                          from_tf=bool('.ckpt' in self.bert_model_path),
                                                          config=self.bert_config)
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.bert_output_mode = bert_output_mode
        self.bert_chinese_word_embedding_select_mode = bert_chinese_word_embedding_select_mode
        if self.bert_output_mode == 'attention':
            self.layer_attention = LayerAttention(self.bert_config.num_hidden_layers, do_layer_norm=False,
                                                  dropout=layer_dropout)
        else:
            self.layer_attention = None
        # debug_print('BERT config:')
        # debug_print(self.config)
        # self.model.to(self.device)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None, end_pos=None, start_pos=None):
        # BERT model output:
        #         **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
        #             Sequence of hidden-states at the output of the last layer of the model.
        #         **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
        #         **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
        #             list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
        #             of shape ``(batch_size, sequence_length, hidden_size)``:
        #             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        #         **attentions**: (`optional`, returned when ``config.output_attentions=True``)
        bert_outputs = self.bert(input_ids, position_ids=position_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        last_layer_hidden_state = bert_outputs[0]
        all_layers_hidden_states = bert_outputs[2]
        assert torch.all(all_layers_hidden_states[-1] == last_layer_hidden_state)
        # Which vector works best as a contextualized embedding? http://jalammar.github.io/illustrated-bert/
        if self.bert_output_mode == 'last':
            encoder_output = last_layer_hidden_state
        elif self.bert_output_mode == 'last_four_sum':
            last_four_hidden_states = torch.stack(all_layers_hidden_states[-4:])
            encoder_output = torch.sum(last_four_hidden_states, 0)
        elif self.bert_output_mode == 'last_four_cat':
            raise NotImplementedError(f'do not support this bert_output_mode:{self.bert_output_mode}')
            # torch.cat
        elif self.bert_output_mode == 'all_sum':
            all_hidden_states = torch.stack(all_layers_hidden_states)
            encoder_output = torch.sum(all_hidden_states, 0)
        elif self.bert_output_mode == 'attention':
            all_hidden_states = torch.stack(all_layers_hidden_states)
            encoder_output = self.layer_attention(all_hidden_states, attention_mask)
        else:
            raise Exception('bad bert output mode')
        # mask:
        # encoder_output = attention_mask.unsqueeze(-1).to(encoder_output.dtype) * encoder_output
        output_pad_mask = torch.eq(attention_mask, 0)
        encoder_output = encoder_output.masked_fill(output_pad_mask.unsqueeze(2), 0)
        # 必须先mask然后再index select
        if self.bert_chinese_word_embedding_select_mode == 's':
            encoder_output = batched_index_select(encoder_output, 1, start_pos)
        elif self.bert_chinese_word_embedding_select_mode == 'e':
            encoder_output = batched_index_select(encoder_output, 1, end_pos)
        elif self.bert_chinese_word_embedding_select_mode == 's+e':
            encoder_output = batched_index_select(encoder_output, 1, start_pos) \
                             + batched_index_select(encoder_output, 1, end_pos)
            encoder_output /= 2
        elif self.bert_chinese_word_embedding_select_mode == 's-e':
            raise NotImplementedError('not support this select mode')
        return self.dropout(encoder_output)


if __name__ == '__main__':
    class Args():
        def __init__(self):
            self.bert_path = '/home/liangs/disk/data/bert-base-chinese'
            self.data_dir = '../dataset'
            self.train_file = 'test.conllu'
            self.max_seq_length = 10
            self.encoder_type = 'bert'
            self.root_representation = 'unused'


    args = Args()

    tokenizer = load_bert_tokenizer('/home/liangs/disk/data/bert-base-chinese', 'bert')
    vocab = GraphVocab('../dataset/graph_vocab.txt')
    dataset, CoNLLU_file = load_and_cache_examples(args, vocab, tokenizer)
    data_loader = get_data_loader(dataset, batch_size=2, evaluation=True)
    bert = BERTTypeEncoder(no_cuda=True, bert_path=args.bert_path)
    for batch in data_loader:
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2] if args.encoder_type in ['bert', 'xlnet'] else None,
            'start_pos': batch[3],
            'end_pos': batch[4],
        }
        # print(inputs)
        # # print(inputs['start_pos'])
        # output = bert(**inputs)
        # print(output)
