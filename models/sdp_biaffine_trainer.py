# -*- coding: utf-8 -*-\
"""
A trainer class to handle training and testing of models.
"""
import torch

# from common.trainer import Trainer as BaseTrainer
from utils.model_utils.optimization import get_optimizer_old
from utils import model_utils
from .hdlstm_biaffine import Parser
from utils.input_utils.common import MultiVocab
# from model_utils.parser_funs import sdp_decoder, parse_semgraph
from utils.model_utils.optimization import BertAdam
from utils.logger import get_logger
from torch.utils.data import (DataLoader, RandomSampler)


class BaseTrainer(object):
    def unpack_batch(self, batch):
        inputs = batch[:6]
        # inputs = [words, words_mark, wordchars, wordchars_mark, upos, pretrained]
        arcs = batch[6]
        orig_idx = batch[7]
        word_orig_idx = batch[8]
        sentlens = batch[9]
        wordlens = batch[10]
        return inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens

    def change_lr(self, new_lr):
        # 更新学习率：
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def save(self, filename):
        # 保存主要参数：
        savedict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(savedict, filename)

    def load(self, filename):
        # 加载参数：
        savedict = torch.load(filename, lambda storage, loc: storage)

        self.model.load_state_dict(savedict['model'])
        if self.args['mode'] == 'train':
            self.optimizer.load_state_dict(savedict['optimizer'])


class Trainer(BaseTrainer):
    """ A trainer for training models. """

    # eg: trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'])
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(pretrain, model_file)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Parser(args, vocab, emb_matrix=pretrain.emb)
            self.logger = get_logger(args['logger_name'])
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        if self.args['optim'] == 'BertAdam':

            # 对 bias、gamma、beta变量不使用权重衰减
            # 权重衰减是一种正则化手段
            self.parameters = [p for p in self.model.named_parameters()]
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.parameters if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.parameters if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=self.args['bert_adam_lr'],
                                      warmup=self.args['warmup_proportion'],
                                      t_total=self.args['max_steps'])
        else:
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = get_optimizer_old(self.args['optim'], self.parameters, self.args['lr'],
                                               betas=(self.args['beta1'], self.args['beta2']),
                                               eps=self.args['eps'],
                                               weight_decay=self.args['L2_penalty'])
        # print("------model named parameters:------")
        # for n, p in self.model.named_parameters():
        #     print("name:", n)
        #     print(p)
        # print("---" * 10)
        # print("named para num:",len(list(self.model.named_parameters())))
        # print("para num:",len(list(self.model.parameters())))

    def update(self, batch, global_step, cuda_data=False, eval=False):
        # inputs = [words, words_mark, wordchars, wordchars_mark, upos, pretrained]
        inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda,
                                                                                 is_cuda_tensor=cuda_data)
        word, word_mask, wordchars, wordchars_mask, upos, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, pretrained, arcs, word_orig_idx,
                             sentlens, wordlens)
        if self.args['split_loss']:
            loss_val = (loss[0] + loss[1] + loss[2]).data.item()
        else:
            loss_val = loss.data.item()
        if eval:
            return loss_val
        if self.args['big_batch']:
            loss = loss / self.args['accumulation_steps']
        if self.args['split_loss'] and self.args['nlpcc']:
            loss[0].backward(retain_graph=True)
            loss[1].backward(retain_graph=True)
            loss[2].backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        if self.args['big_batch']:
            if global_step % self.args['accumulation_steps']:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss_val

    def predict(self, batch, cuda_data=False, unsort=True):
        inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda,
                                                                                 is_cuda_tensor=cuda_data)
        # print(sentlens)
        word, word_mask, wordchars, wordchars_mask, upos, pretrained = inputs
        self.model.eval()

        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, pretrained, arcs, word_orig_idx,
                              sentlens, wordlens)

        semgraph = model_utils.sdp_decoder(preds[0], sentlens)
        sents = model_utils.parse_semgraph(semgraph, sentlens)
        pred_sents = self.vocab['graph'].parse_to_sent_batch(sents)
        if unsort:
            pred_sents = model_utils.unsort(pred_sents, orig_idx)
        return pred_sents

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
            'model': model_state,
            'vocab': self.vocab.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
            self.logger.info("model saved to {}".format(filename))
        except BaseException:
            self.logger.exception("[Warning: Saving failed... continuing anyway.]")

    def load(self, pretrain, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            self.logger.exception("Cannot load model from {}".format(filename))
            exit()
        self.args = checkpoint['config']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = Parser(self.args, self.vocab, emb_matrix=pretrain.emb)
        self.model.load_state_dict(checkpoint['model'], strict=False)


class BERTBiaffineTrainer():
    def __init__(self, model):
        self.model = model

    def train(self, args, train_dataset):
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)


if __name__ == '__main__':
    pass
