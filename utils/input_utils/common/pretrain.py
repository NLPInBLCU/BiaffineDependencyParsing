"""
Supports for pretrained data.
"""
import numpy as np
import pickle

from .base_vocab import BaseVocab, VOCAB_PREFIX
from utils.logger import get_logger


class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        # VOCAB_PREFIX = [PAD, UNK, EMPTY, ROOT]
        # self.data: words list
        self._id2unit = VOCAB_PREFIX + self.data
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}


class Pretrain(object):
    """ A loader and saver for pretrained embeddings.
    eg:
            vec_file = './Embeds/sdp_vec.pkl'
            pretrain_file = './save/sdp.pretrain.pt'
            # 传入torch预训练的模型文件（pt）和Python二进制持久化文件（pkl）
            pretrain = Pretrain(pretrain_file, vec_file)
    """

    def __init__(self, vec_filename, logger_name=__name__):
        # self.filename = filename
        self.vec_filename = vec_filename
        self.logger = get_logger(logger_name)

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self._vocab, self._emb = self.load()
        return self._vocab

    @property
    def emb(self):
        if not hasattr(self, '_emb'):
            self._vocab, self._emb = self.load()
        return self._emb

    def load(self):
        # if os.path.exists(self.filename):
        #     try:
        #         # 加载预训练torch文件
        #         data = torch.load(self.filename, lambda storage, loc: storage)
        #     except BaseException as e:
        #         self.logger.exception(
        #             "Pretrained file exists but cannot be loaded from {}, due to the following exception:".format(
        #                 self.filename))
        #         # print("\t{}".format(e))
        #         return self.read_and_save_hit()
        #     # 返回预训练的vocab和emb（字典和词向量矩阵）
        #     assert len(data['vocab']) == len(data['emb'])
        #     return data['vocab'], data['emb']
        # else:
        #     return self.read_and_save_hit()
        return self.read_and_save_hit()

    # def read_and_save(self):
    #     # load from pretrained filename
    #     if self.vec_filename is None:
    #         raise Exception("Vector file is not provided.")
    #     self.logger.info("Reading pretrained vectors from {}...".format(self.vec_filename))
    #     first = True
    #     words = []
    #     failed = 0
    #     with lzma.open(self.vec_filename, 'rb') as f:
    #         for i, line in enumerate(f):
    #             # print(line)
    #             try:
    #                 line = line.decode()
    #             except UnicodeDecodeError:
    #                 failed += 1
    #                 continue
    #             if first:
    #                 # the first line contains the number of word vectors and the dimensionality
    #                 first = False
    #                 line = line.strip().split(' ')
    #                 rows, cols = [int(x) for x in line]
    #                 emb = np.zeros((rows + len(VOCAB_PREFIX), cols), dtype=np.float32)
    #                 continue
    #
    #             line = line.rstrip().split(' ')
    #             emb[i + len(VOCAB_PREFIX) - 1 - failed, :] = [float(x) for x in line[-cols:]]
    #             words.append(' '.join(line[:-cols]))
    #
    #     vocab = PretrainedWordVocab(words, lower=True)
    #
    #     if failed > 0:
    #         emb = emb[:-failed]
    #
    #     # save to file
    #     data = {'vocab': vocab, 'emb': emb}
    #     try:
    #         torch.save(data, self.filename)
    #         self.logger.critical("Saved pretrained vocab and vectors to {}".format(self.filename))
    #     except BaseException as e:
    #         self.logger.exception("Saving pretrained data failed due to the following exception... continuing anyway")
    #         # print("\t{}".format(e))
    #
    #     return vocab, emb

    def read_and_save_hit(self):
        self.logger.critical(f'Use vec_file: {self.vec_filename}')
        # 如果self.filename不存在：
        if self.vec_filename is None:
            raise Exception("Vector file is not provided.")
        self.logger.critical("Reading pretrained vectors from {}...".format(self.vec_filename))
        with open(self.vec_filename, 'rb') as f:
            result = pickle.load(f)
            orig_vocab, orig_emb = result
            # vocab:list; emb:numpy array
            rows, cols = orig_emb.shape
            words = orig_vocab
            vocab = PretrainedWordVocab(words, lower=True)
            emb = np.zeros((rows + len(VOCAB_PREFIX), cols), dtype=np.float32)
            for i in range(rows):
                emb[len(VOCAB_PREFIX) + i] = orig_emb[i]
            data = {'vocab': vocab, 'emb': emb}
        # try:
        #     torch.save(data, self.filename)
        #     self.logger.critical("Saved pretrained vocab and vectors to {}".format(self.filename))
        # except BaseException as e:
        #     self.logger.exception("Saving pretrained data failed due to the following exception... continuing anyway")
        # print("\t{}".format(e))
        # 返回预训练的vocab和emb（字典和词向量矩阵）
        assert len(data['vocab']) == len(data['emb'])
        return data['vocab'], data['emb']


if __name__ == '__main__':
    from pprint import pprint

    # filename = '../save/sdp.pretrain.pt'
    vec_filename = '../Embeds/sdp_correct_embeds.pkl'
    pretrain = Pretrain(vec_filename)
    vocab = pretrain.vocab
    embeds = pretrain.emb
    print(embeds[278])
    print(type(vocab))
    pprint(dir(vocab))
