# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/28
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

from utils.information import debug_print
from utils.input_utils.conll_file import CoNLLFile
from utils.input_utils.graph_vocab import GraphVocab
from utils.model_utils.get_optimizer import get_optimizer
from utils.model_utils.parser_funs import sdp_decoder, parse_semgraph
import utils.model_utils.sdp_simple_scorer as sdp_scorer
from utils.best_result import BestResult
from utils.seed import set_seed
from utils.model_utils.label_smoothing import label_smoothed_kl_div_loss


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class BiaffineDependencyTrainer(metaclass=ABCMeta):
    def __init__(self, args, model):
        self.model = model
        self.optimizer, self.optim_scheduler = get_optimizer(args, model)
        self.graph_vocab = GraphVocab(args.graph_vocab_file)
        self.args = args

    @abstractmethod
    def _unpack_batch(self, args, batch):
        """
            拆分batch，得到encoder的输入和word mask，sentence length，以及dep ids
        :param args: 配置参数
        :param batch: 输入的单个batch,类型为TensorDataset(或者torchtext.dataset)，可用索引分别取值
        :return:返回一个元祖，[1]是inputs，类型为字典；[2]是word mask；[3]是sentence length,python 列表；[4]是dep ids
        """
        raise NotImplementedError('must implement in sub class')

    def _update_and_predict(self, unlabeled_scores, labeled_scores, unlabeled_target, labeled_target, word_pad_mask,
                            label_loss_ratio=None, sentence_lengths=None,
                            calc_loss=True, update=True, calc_prediction=False):
        """
            针对一个batch输入：计算loss，反向传播，计算预测结果
            :param word_pad_mask: 以word为单位，1为PAD，0为真实输入
        :return:
        """
        weights = torch.ones(word_pad_mask.size(0), self.args.max_seq_len, self.args.max_seq_len,
                             dtype=unlabeled_scores.dtype,
                             device=unlabeled_scores.device)
        # 将PAD的位置权重设为0，其余位置为1
        weights = weights.masked_fill(word_pad_mask.unsqueeze(1), 0)
        weights = weights.masked_fill(word_pad_mask.unsqueeze(2), 0)
        # words_num 记录batch中的单词数量
        # torch.eq(word_pad_mask, False) 得到word_mask
        words_num = torch.sum(torch.eq(word_pad_mask, False)).item()
        if calc_loss:
            assert label_loss_ratio
            assert unlabeled_target is not None and labeled_target is not None
            crit_head = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
            head_loss = crit_head(unlabeled_scores, unlabeled_target)

            crit_rel = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
            dependency_mask = labeled_target.eq(0)
            labeled_target = labeled_target.masked_fill(dependency_mask, -1)
            labeled_scores = labeled_scores.contiguous().view(-1, len(self.graph_vocab.get_labels()))
            rel_loss = crit_rel(labeled_scores, labeled_target.view(-1))

            loss = 2 * ((1 - label_loss_ratio) * head_loss + label_loss_ratio * rel_loss)

            if self.args.average_loss_by_words_num:
                loss = loss / words_num

            if self.args.scale_loss:
                loss = loss * self.args.loss_scaling_ratio

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if update:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if self.optim_scheduler:
                    self.optim_scheduler.step()  # Update learning rate schedule
                self.optimizer.step()
                self.model.zero_grad()
            loss = loss.detach().cpu().item()
        else:
            loss = None
        if calc_prediction:
            assert sentence_lengths
            weights = weights.unsqueeze(3)
            head_probs = torch.sigmoid(unlabeled_scores).unsqueeze(3)
            label_probs = torch.softmax(labeled_scores, dim=3)
            batch_probs = head_probs * label_probs * weights
            batch_probs = batch_probs.detach().cpu().numpy()
            # debug_print(batch_probs)
            sem_graph = sdp_decoder(batch_probs, sentence_lengths)
            sem_sents = parse_semgraph(sem_graph, sentence_lengths)
            batch_prediction = self.graph_vocab.parse_to_sent_batch(sem_sents)
        else:
            batch_prediction = None
        return loss, batch_prediction

    def train(self, train_data_loader, dev_data_loader=None, dev_CoNLLU_file=None):
        global_step = 0
        best_result = BestResult()
        self.model.zero_grad()
        set_seed(self.args)  # Added here for reproductibility (even between python 2 and 3)
        train_stop = False
        summary_writer = SummaryWriter(log_dir=self.args.summary_dir)
        for epoch in range(1, self.args.max_train_epochs + 1):
            epoch_ave_loss = 0
            train_data_loader = tqdm(train_data_loader, desc=f'Training epoch {epoch}')
            for step, batch in enumerate(train_data_loader):
                batch = tuple(t.to(self.args.device) for t in batch)
                self.model.train()
                # debug_print(batch)
                # word_mask:以word为单位，1为真实输入，0为PAD
                inputs, word_mask, _, dep_ids = self._unpack_batch(self.args, batch)
                # word_pad_mask:以word为单位，1为PAD，0为真实输入
                word_pad_mask = torch.eq(word_mask, 0)
                unlabeled_scores, labeled_scores = self.model(inputs)
                labeled_target = dep_ids
                unlabeled_target = labeled_target.ge(1).to(unlabeled_scores.dtype)
                # Calc loss and update:
                loss, _ = self._update_and_predict(unlabeled_scores, labeled_scores, unlabeled_target, labeled_target,
                                                   word_pad_mask,
                                                   label_loss_ratio=self.model.label_loss_ratio if not self.args.parallel_train else self.model.module.label_loss_ratio,
                                                   calc_loss=True, update=True, calc_prediction=False)
                global_step += 1
                if loss is not None:
                    epoch_ave_loss += loss

                if global_step % self.args.eval_interval == 0:
                    summary_writer.add_scalar('loss/train', loss, global_step)
                    if dev_data_loader:
                        UAS, LAS = self.dev(dev_data_loader, dev_CoNLLU_file)
                        summary_writer.add_scalar('metrics/uas', UAS, global_step)
                        summary_writer.add_scalar('metrics/las', LAS, global_step)
                        if best_result.is_new_record(LAS=LAS, UAS=UAS, global_step=global_step):
                            print(f"\n## NEW BEST RESULT in epoch {epoch} ##")
                            print(best_result)

                if self.args.early_stop and global_step - best_result.best_LAS_step > self.args.early_stop_steps:
                    print(f'\n## Early stop in step:{global_step} ##')
                    train_stop = True
                    break
            if train_stop:
                break
            # print(f'\n- Epoch {epoch} average loss : {epoch_ave_loss / len(train_data_loader)}')
            summary_writer.add_scalar('epoch_loss', epoch_ave_loss / len(train_data_loader), epoch)
        with open(self.args.dev_result_path, 'w', encoding='utf-8')as f:
            f.write(str(best_result) + '\n')
        print("\n## BEST RESULT in Training ##")
        print(best_result)
        summary_writer.close()

    def dev(self, dev_data_loader, dev_CoNLLU_file):
        assert isinstance(dev_CoNLLU_file, CoNLLFile)
        dev_data_loader = tqdm(dev_data_loader, desc='Evaluation')
        predictions = []
        # batch_sent_lens = []
        for step, batch in enumerate(dev_data_loader):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs, word_mask, sent_lens, dep_ids = self._unpack_batch(self.args, batch)
            word_mask = torch.eq(word_mask, 0)
            unlabeled_scores, labeled_scores = self.model(inputs)
            try:
                with torch.no_grad():
                    _, batch_prediction = self._update_and_predict(unlabeled_scores, labeled_scores, None, None,
                                                                   word_mask,
                                                                   label_loss_ratio=self.model.label_loss_ratio if not self.args.parallel_train else self.model.module.label_loss_ratio,
                                                                   sentence_lengths=sent_lens,
                                                                   calc_loss=False, update=False, calc_prediction=True)
            except Exception as e:
                for b in batch:
                    print(b.shape)
                raise e
            predictions += batch_prediction
            # batch_sent_lens += sent_lens

        dev_CoNLLU_file.set(['deps'], [dep for sent in predictions for dep in sent])
        dev_CoNLLU_file.write_conll(self.args.dev_output_path)
        UAS, LAS = sdp_scorer.score(self.args.dev_output_path, os.path.join(self.args.data_dir, self.args.dev_file))
        return UAS, LAS

    def inference(self, inference_data_loader, inference_CoNLLU_file):
        inference_data_loader = tqdm(inference_data_loader, desc='Inference')
        predictions = []
        for step, batch in enumerate(inference_data_loader):
            self.model.eval()
            inputs, word_mask, sent_lens, _ = self._unpack_batch(self.args, batch)
            word_mask = torch.eq(word_mask, 0)
            unlabeled_scores, labeled_scores = self.model(inputs)
            with torch.no_grad():
                _, batch_prediction = self._update_and_predict(unlabeled_scores, labeled_scores, None, None, word_mask,
                                                               label_loss_ratio=self.model.label_loss_ratio if not self.args.parallel_train else self.model.module.label_loss_ratio,
                                                               sentence_lengths=sent_lens,
                                                               calc_loss=False, update=False, calc_prediction=True)
            predictions += batch_prediction
        inference_CoNLLU_file.set(['deps'], [dep for sent in predictions for dep in sent])
        inference_CoNLLU_file.write_conll(self.args.test_output_path)
        return predictions


class BERTBiaffineTrainer(BiaffineDependencyTrainer):
    def _unpack_batch(self, args, batch):
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2] if args.encoder_type in ['bert', 'xlnet'] else None,
            'start_pos': batch[3],
            'end_pos': batch[4],
        }
        dep_ids = batch[5]
        # word_mask:以word为单位，1为真实输入，0为PAD
        word_mask = (batch[3] != (args.max_seq_len - 1)).to(torch.long).to(args.device)
        sent_len = torch.sum(word_mask, 1).cpu().tolist()
        return inputs, word_mask, sent_len, dep_ids


class TransformerBiaffineTrainer(BiaffineDependencyTrainer):
    def _unpack_batch(self, args, batch):
        pass


class CharRNNBiaffineTrainer(BiaffineDependencyTrainer):
    pass


if __name__ == '__main__':
    pass
