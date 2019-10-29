# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_wrapper
   Description :
   Author :       Liangs
   date：          2019/7/27
-------------------------------------------------
   Change Activity:
                   2019/7/27:
-------------------------------------------------
"""
from abc import ABCMeta, abstractmethod
from utils.logger import *
from utils import model_utils, model_utils as sdp_scorer
from utils.model_utils.optimization import *
from utils.best_result import BestResult


class Wrapper(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def unpack_batch(self, batch: list):
        """
        
        :param batch:
        batch=(
            words,  0
            words_mask,     1
            wordchars,      2
            wordchars_mask,     3
            upos,       4
            pretrained,     5
            arcs,       6
            orig_idx,       7
            word_orig_idx,      8
            sentlens,       9
            word_lens       10
          ) 
        :return: 
        """
        pass

    def train(self, train_iter, dev_iter):
        global_step = 0
        train_loss = 0
        using_amsgrad = False
        best_result = BestResult()
        for epoch in range(1, self.args.max_epochs):
            do_break = False
            for i, batch in enumerate(train_iter):
                global_step += 1
                loss = self._update(batch, global_step, eval=False)  # update step
                train_loss += loss
                if self.args.cuda and self.args.empty_cache:
                    torch.cuda.empty_cache()
                if global_step % self.args['log_step'] == 0:
                    self.summary_writer.add_scalar('loss', loss, global_step)
                if global_step % self.args['eval_interval'] == 0:
                    train_loss = train_loss / self.args['eval_interval']  # avg loss per batch
                    self._evaluate(dev_iter, global_step, best_result)
                if global_step - max(last_best_step) >= self.args['max_steps_before_stop']:
                    if not using_amsgrad and self.args.using_amsgard:
                        logger.critical(f"--->>> Optim:Switching to AMSGrad in step:{global_step}")
                        last_best_step = [global_step] * 2
                        using_amsgrad = True
                        # todo: amsgrad:
                        # self.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'],
                        #                             betas=(args['beta1'], args['beta2']), eps=args['eps'])
                    else:
                        do_break = True
                        break
                if global_step >= self.args['max_steps']:
                    do_break = True
                    break
            if do_break:
                break
            train_iter.reshuffle()

    def _evaluate(self, dev_iter, step, best_result):
        dev_preds = []
        for batch in dev_iter:
            preds = self._predict(batch)
            dev_preds += preds
            if self.args.cuda and self.args.empty_cache:
                torch.cuda.empty_cache()
        dev_iter.conll.set(['deps'], [y for x in dev_preds for y in x])
        dev_iter.conll.write_conll(self.args.dev_output_file)
        dev_uas, dev_las = sdp_scorer.score(self.args.dev_output_file, self.args.dev_gold_file)
        self.summary_writer.add_scalars(f'data/dev',
                                        {
                                            'dev_UAS': dev_uas,
                                            'dev_LAS': dev_las,
                                        },
                                        step)
        # save best model
        if best_result.is_new_record(LAS=dev_las, UAS=dev_uas, global_step=step):
            self.save(self.args.save_model_file)

    def inference(self, test_iter):
        if len(test_iter) > 0:
            print(f"Start evaluation...")
            preds = []
            for i, batch in enumerate(test_iter):
                preds += self._predict(batch)
        else:
            preds = []
        test_iter.conll.set(['deps'], [y for x in preds for y in x])
        test_iter.conll.write_conll(self.args.test_output_file)
        if self.args.test_gold_file is not None:
            UAS, LAS = sdp_scorer.score(self.args.test_output_file, self.args.test_gold_file)
            print(f"Test Dataset Parser score:")
            print(f'LAS:{LAS * 100:.3f}\tUAS:{UAS * 100:.3f}')

    @abstractmethod
    def _update(self, batch, global_step, eval=False):
        pass

    @abstractmethod
    def _predict(self, batch, unsort=True):
        pass

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings,
        # because they are large and will be saved in a separate file
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

    @abstractmethod
    def load(self, saved_model_file):
        pass


class LSTMModelWrapper(Wrapper):
    def __init__(self, args, model=None):
        if args.saved_model_file and args.continue_train:
            self.load(args.saved_model_file)
        else:
            if not model:
                raise ValueError('model can not be None')
            self.model = model
            self.args = args
        self.logger = get_logger(args.logger_name)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = get_optimizer_old(self.args.optim, self.parameters, self.args.lr,
                                           betas=(self.args.beta1, self.args.beta2),
                                           eps=self.args.eps,
                                           weight_decay=self.args.L2_penalty)
        self.model_class = args.model_class
        self.pretrain = args.pretrain
        self.vocab = args.vocab
        super().__init__()

    def unpack_batch(self, batch: list):
        inputs = batch[:6]
        # inputs = [words, words_mark, wordchars, wordchars_mark, upos, pretrained]
        arcs = batch[6]
        orig_idx = batch[7]
        word_orig_idx = batch[8]
        sentlens = batch[9]
        wordlens = batch[10]
        return inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens

    def load(self, saved_model_file):
        try:
            checkpoint = torch.load(saved_model_file, lambda storage, loc: storage)
        except BaseException as e:
            raise e
        self.args = checkpoint['config']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = self.model_class(self.args, self.vocab, emb_matrix=self.pretrain.emb)
        self.model.load_state_dict(checkpoint['model'], strict=False)

    def _update(self, batch, global_step, eval=False):
        # inputs = [words, words_mark, wordchars, wordchars_mark, upos, pretrained]
        inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens = self.unpack_batch(batch)
        word, word_mask, wordchars, wordchars_mask, upos, pretrained = inputs
        if eval:
            self.model.eval()
        else:
            self.model.train()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, pretrained, arcs, word_orig_idx,
                             sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val
        if self.args.accumulation:
            loss = loss / self.args.accumulation_steps
        loss.backward()
        if self.args.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        if self.args.accumulation:
            if global_step % self.args.accumulation_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss_val

    def _predict(self, batch, unsort=True):
        inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens = self.unpack_batch(batch)
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


class TransformerModelWrapper(Wrapper):
    def __init__(self, *kwarg):
        super().__init__(*kwarg)

    def unpack_batch(self, batch: list):
        pass

    def load(self, saved_model_file):
        pass

    def _predict(self, batch, unsort=True):
        pass

    def _update(self, batch, global_step, eval=False):
        pass
