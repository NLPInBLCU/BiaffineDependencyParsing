# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     parser_funs
   Description :
   Author :       shen zizhuo
   date：
-------------------------------------------------
   Change Activity:
                   2019/7/28:
-------------------------------------------------
"""
import torch
import numpy as np


def sdp_decoder(semgraph_probs, sentlens):
    '''
    semhead_probs type:ndarray, shape:(n,m,m)
    '''
    semhead_probs = semgraph_probs.sum(axis=-1)
    semhead_preds = np.where(semhead_probs >= 0.5, 1, 0)
    masked_semhead_preds = np.zeros(semhead_preds.shape, dtype=np.int32)
    for i, (sem_preds, length) in enumerate(zip(semhead_preds, sentlens)):
        masked_semhead_preds[i, :length, :length] = sem_preds[:length, :length]
    n_counts = {'no_root': 0, 'multi_root': 0, 'no_head': 0, 'self_circle': 0}
    for i, length in enumerate(sentlens):
        for j in range(length):
            if masked_semhead_preds[i, j, j] == 1:
                n_counts['self_circle'] += 1
                masked_semhead_preds[i, j, j] = 0
        n_root = np.sum(masked_semhead_preds[i, :, 0])
        if n_root == 0:
            n_counts['no_root'] += 1
            new_root = np.argmax(semhead_probs[i, 1:, 0]) + 1
            masked_semhead_preds[i, new_root, 0] = 1
        elif n_root > 1:
            n_counts['multi_root'] += 1
            kept_root = np.argmax(semhead_probs[i, 1:, 0]) + 1
            masked_semhead_preds[i, :, 0] = 0
            masked_semhead_preds[i, kept_root, 0] = 1
        n_heads = masked_semhead_preds[i, :length, :length].sum(axis=-1)
        n_heads[0] = 1
        for j, n_head in enumerate(n_heads):
            if n_head == 0:
                n_counts['no_head'] += 1
                semhead_probs[i, j, j] = 0
                new_head = np.argmax(semhead_probs[i, j, 1:length]) + 1
                masked_semhead_preds[i, j, new_head] = 1
    # (n x m x m x c) -> (n x m x m)
    semrel_preds = np.argmax(semgraph_probs, axis=-1)
    # (n x m x m) (*) (n x m x m) -> (n x m x m)
    semgraph_preds = masked_semhead_preds * semrel_preds
    result = masked_semhead_preds + semgraph_preds
    return result


def parse_semgraph(semgraph, sentlens):
    semgraph = semgraph.tolist()
    sents = []
    for s, l in zip(semgraph, sentlens):
        words = []
        for w in s[1:l]:
            arc = []
            for head_idx, deprel in enumerate(w[:l]):
                if deprel == 0:
                    continue
                arc.append([head_idx, deprel - 1])
            words.append(arc)
        sents.append(words)
    return sents
