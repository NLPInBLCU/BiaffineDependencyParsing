# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     make_target
   Description :
   Author :       Liangs
   date：          2019/7/28
-------------------------------------------------
   Change Activity:
                   2019/7/28:
-------------------------------------------------
"""
import torch
from utils.input_utils.deprecated_common import GraphVocab
from utils.input_utils.conll_file import load_conllu_file


def make_unlabeltarget(arcs, sentlens, use_cuda=False):
    max_len = sentlens[0]
    # print(max_len)
    batch_size = len(arcs)
    # print(batch_size)
    graphs = torch.zeros(batch_size, max_len, max_len)
    sent_idx = 0
    for sent in arcs:
        word_idx = 1
        for word in sent:
            for arc in word:
                # print(sent_idx, word_idx, arc)
                head_idx = arc[0]
                graphs[sent_idx, word_idx, head_idx] = 1
            word_idx += 1
        sent_idx += 1
    if use_cuda:
        graphs = graphs.float().cuda()
    else:
        graphs = graphs.float()
    return graphs


def make_labeltarget(arcs, sentlens, use_cuda=False):
    max_len = sentlens[0]
    # print(max_len)
    batch_size = len(arcs)
    # print(batch_size)
    graphs = torch.zeros(batch_size, max_len, max_len)
    # print(graphs.shape)
    sent_idx = 0
    for sent in arcs:
        word_idx = 1
        for word in sent:
            for arc in word:
                # print(sent_idx, word_idx, arc)
                head_idx = arc[0]
                rel_idx = arc[1]
                graphs[sent_idx, word_idx, head_idx] = rel_idx
            word_idx += 1
        sent_idx += 1
    if use_cuda:
        graphs = graphs.long().cuda()
    else:
        graphs = graphs.long()
    return graphs


def make_discriminator_target(sent_num, task_id, use_cuda=False):
    labels = torch.zeros(sent_num)
    labels.fill_(task_id)
    if use_cuda:
        labels = labels.long().cuda()
    else:
        labels = labels.long()
    return labels


if __name__ == '__main__':

    def make_label_target(arcs, max_seq_length):
        graphs = [[0] * max_seq_length for _ in range(max_seq_length)]
        for word_idx, word in enumerate(arcs, start=1):
            for arc in word:
                head_idx = arc[0]
                rel_idx = arc[1]
                graphs[word_idx][head_idx] = rel_idx
        return graphs


    file_name = "../dataset/test/sdp_text_test.conllu"
    conllu_file, data = load_conllu_file(file_name)
    data = data[:3]
    vocab = GraphVocab(data, idx=2)
    sent = data[0]
    sentlens = [len(sent) + 1]  # +1 for ROOT
    arcs = [vocab.get_arc(sent, 2)]
    # for s, a in zip(sent, arcs[0]):
    #     print(s, a)
    ut = make_unlabeltarget(arcs, sentlens, False)
    lt = make_labeltarget(arcs, sentlens, False)
    # print(lt.dtype)
    # print(ut)
    # uut = lt.ge(1).to(lt.dtype)
    # print(torch.equal(ut.float(), uut.float()))
    print(lt)
    print(torch.tensor(make_label_target(arcs[0], sentlens[0]), dtype=lt.dtype))
