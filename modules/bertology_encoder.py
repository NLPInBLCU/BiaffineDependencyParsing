# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24
import torch
import torch.nn as nn

from pytorch_transformers import (BertConfig,
                                  BertTokenizer,
                                  BertModel)

from modules.layer_attention import LayerAttention
from modules.transformer_layer import TransformerSentenceEncoderLayer

BERTology_MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    # 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # 'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # 'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def batched_index_select(t, dim, inds):
    """
        实现batch版本的index select
    :param t:
    :param dim:
    :param inds:
    :return:
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


class BERTologyEncoder(nn.Module):
    def __init__(
            self,
            no_cuda=False,
            bertology='bert',
            bertology_path='.',
            bertology_output_mode='last_four_sum',
            bertology_word_select_mode='s',
            layer_dropout=0.1,
            bertology_after='none',
            after_layers=0,
            max_seq_len=None,
            after_dropout=0.1,
            **kwargs,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()
        assert max_seq_len is not None, "必须指定max_seq_len"
        self.max_seq_len = max_seq_len
        self.bertology_type = bertology.lower()
        assert self.bertology_type in BERTology_MODEL_CLASSES.keys(), \
            f'BERTology Only support {list(BERTology_MODEL_CLASSES.keys())}'
        self.bertology_path = bertology_path
        self.bertology_config_class, self.bertology_model_class, _ = BERTology_MODEL_CLASSES[self.bertology_type]
        self.bertology_config = self.bertology_config_class.from_pretrained(self.bertology_path)
        self.bertology_config.output_hidden_states = True
        # 注意这里不加载BERT的预训练参数
        # BERT的参数通过Model.from_pretrained方法加载
        self.bertology = self.bertology_model_class(config=self.bertology_config)
        self.dropout = nn.Dropout(self.bertology_config.hidden_dropout_prob)
        self.bertology_output_mode = bertology_output_mode
        self.bertology_word_select_mode = bertology_word_select_mode
        if self.bertology_output_mode == 'attention':
            self.layer_attention = LayerAttention(self.bertology_config.num_hidden_layers, do_layer_norm=False,
                                                  dropout=layer_dropout)
        else:
            self.layer_attention = None
        if bertology_after.lower() != 'none':
            assert bertology_after.lower() in ['transformer']
            assert after_layers > 0
        if bertology_after.lower() == 'transformer':
            self.after_encoder = nn.ModuleList(
                [
                    TransformerSentenceEncoderLayer(embedding_dim=self.bertology_config.hidden_size,
                                                    dropout=after_dropout)
                    for _ in range(after_layers)
                ]
            )
        else:
            self.after_encoder = None

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None, end_pos=None, start_pos=None):
        bert_outputs = self.bertology(input_ids, position_ids=position_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask, head_mask=head_mask)
        # bert_outputs: 0: sequence_output, 1: pooled_output, 2:tuple (hidden_states), 3:tuple (attentions)
        last_layer_hidden_state = bert_outputs[0]
        # hidden_states:one for the output of each layer + the output of the embeddings
        all_layers_hidden_states = bert_outputs[2][1:]
        # Which vector works best as a contextualized embedding? http://jalammar.github.io/illustrated-bertology/
        if self.bertology_output_mode == 'last':
            encoder_output = last_layer_hidden_state
        elif self.bertology_output_mode == 'last_four_sum':
            last_four_hidden_states = torch.stack(all_layers_hidden_states[-4:])
            encoder_output = torch.sum(last_four_hidden_states, 0)
        elif self.bertology_output_mode == 'all_sum':
            all_hidden_states = torch.stack(all_layers_hidden_states)
            encoder_output = torch.sum(all_hidden_states, 0)
        elif self.bertology_output_mode == 'attention':
            # EMNLP paper: 75 Languages, 1 Model: Parsing Universal Dependencies Universally
            encoder_output = self.layer_attention(all_layers_hidden_states, attention_mask)
        else:
            raise Exception('bad bertology output mode')
        # mask:
        # encoder_output = attention_mask.unsqueeze(-1).to(encoder_output.dtype) * encoder_output
        output_pad_mask = torch.eq(attention_mask, 0)
        encoder_output = encoder_output.masked_fill(output_pad_mask.unsqueeze(2), 0)
        # 必须先mask然后再index select
        if self.bertology_word_select_mode == 's':
            encoder_output = batched_index_select(encoder_output, 1, start_pos)
        elif self.bertology_word_select_mode == 'e':
            encoder_output = batched_index_select(encoder_output, 1, end_pos)
        elif self.bertology_word_select_mode == 's+e':
            encoder_output = batched_index_select(encoder_output, 1, start_pos) \
                             + batched_index_select(encoder_output, 1, end_pos)
            encoder_output /= 2
        elif self.bertology_word_select_mode == 's-e':
            raise NotImplementedError('not support this select mode')
        # transformer输入需要attention pad，也就是需要指出哪些是pad的输入
        # 注意这里不能直接使用attention mask作为transformer的输入，这是因为attention mask是原来的字序列的mask
        # 这里我们需要词序列的mask：
        word_attention_pad_mask = torch.eq(start_pos, self.max_seq_len - 1)
        # 确保pad位置向量为0
        encoder_output *= (1 - word_attention_pad_mask.unsqueeze(-1).type_as(encoder_output))
        if self.after_encoder is not None:
            # batch X Seq_len X dim -> Seq_len X batch X dim
            encoder_output = encoder_output.transpose(0, 1)
            for layer in self.after_encoder:
                encoder_output, _ = layer(encoder_output, self_attn_padding_mask=word_attention_pad_mask)
            # Seq_len X batch X dim -> batch X Seq_len X dim
            encoder_output = encoder_output.transpose(0, 1)
        return self.dropout(encoder_output)
