# -*- coding: utf-8 -*-
# Created by li huayong on 2020/1/6
import re
import torch
from trainers.base_trainer import BaseDependencyTrainer


class BERTologyBaseTrainer(BaseDependencyTrainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.freeze:
            assert config.freeze_bertology_layers >= -1 and config.freeze_epochs in ['all', 'first']
        self._freeze = config.freeze
        self._freeze_layers = config.freeze_bertology_layers
        self._freeze_epochs = config.freeze_epochs
        # 记录被freeze的参数名称
        # 在第一个epoch时刻写入
        # 之后直接读取这个list中的参数
        self._freeze_parameter_names = []

    def _unpack_batch(self, args, batch):
        """
            dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_start_pos,
                            all_end_pos, all_dep_ids,
                            all_pos_ids)
        :param args:
        :param batch:
        :return:
        """
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],  # 默认 1 代表 实际输入； 0 代表 padding
            'token_type_ids': batch[2] if args.encoder_type in ['bertology', 'xlnet'] else None,
            'start_pos': batch[3],
            'end_pos': batch[4],
        }
        dep_ids = batch[5]
        pos_ids = batch[6]
        # word_mask:以word为单位，1为真实输入，0为PAD
        word_mask = (batch[3] != (args.max_seq_len - 1)).to(torch.long).to(args.device)
        sent_len = torch.sum(word_mask, 1).cpu().tolist()
        unpacked_batch = {
            'inputs': inputs,
            'word_mask': word_mask,
            'sent_len': sent_len,
            'dep_ids': dep_ids,
            'pos_ids': pos_ids,
        }
        return unpacked_batch

    def _custom_train_operations(self, epoch):
        """
            重写BiaffineDependencyTrainer的_custom_train_operations方法，
            提供定制化的训练操作
            这里我们做BERTology中某些层的freeze
        :param epoch:
        :return:
        """
        if self._freeze:
            if epoch == 1:
                # 首个epoch一定会freeze
                # 利用正则识别参数名，对其进行freeze
                for name, para in self.model.named_parameters():
                    # Freeze BERTology embedding layer
                    if 'encoder.bertology.embeddings.' in name:
                        para.requires_grad = False
                        self._freeze_parameter_names.append(name)
                    # Freeze other BERTology Layers
                    # 如果self._freeze_layers > -1；则说明除了freeze embedding层之外，还要freeze别的层
                    if self._freeze_layers > -1:
                        if re.match(
                                f'^.+encoder\.bertology\.encoder\.layer\.[0-{self._freeze_layers}]\..+',
                                name):
                            para.requires_grad = False
                        self._freeze_parameter_names.append(name)
            else:
                # 如果_freeze_epochs == ‘all’,则此时不需要做任何事情，因为第一个epoch已经freeze过了
                if self._freeze_epochs == 'first':
                    # 仅在第一个epoch freeze参数
                    # epoch大于1的时候不再需要freeze了
                    # 此时将_freeze_parameter_names中的参数重新设置为需要计算梯度
                    for name, para in self.model.named_parameters():
                        if name in self._freeze_parameter_names:
                            para.requires_grad = True


if __name__ == '__main__':
    pass
