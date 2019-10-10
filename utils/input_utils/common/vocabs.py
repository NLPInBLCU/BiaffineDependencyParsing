# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 23:47:28 2019

@author: mypc
"""
from collections import Counter, OrderedDict
from .base_vocab import BaseVocab, BaseMultiVocab
from .base_vocab import VOCAB_PREFIX, EMPTY, EMPTY_ID
from pprint import pprint


class CharVocab(BaseVocab):
    # eg:
    #    charvocab = CharVocab(data, self.args['shorthand'])
    def build_vocab(self):
        # word : line[0]
        counter = Counter([c for sent in self.data for line in sent for c in line[self.idx]])

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}


class WordVocab(BaseVocab):
    # eg:
    #     wordvocab = WordVocab(data, self.args['shorthand'], cutoff=0, lower=True)
    #     uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
    def __init__(self, data=None, lang="", idx=0, cutoff=0, lower=False, ignore=[]):
        self.ignore = ignore
        super().__init__(data, lang=lang, idx=idx, cutoff=cutoff, lower=lower)
        self.state_attrs += ['ignore']

    def id2unit(self, id):
        if len(self.ignore) > 0 and id == EMPTY_ID:
            return '_'
        else:
            return super().id2unit(id)

    def unit2id(self, unit):
        if len(self.ignore) > 0 and unit in self.ignore:
            return self._unit2id[EMPTY]
        else:
            return super().unit2id(unit)

    def build_vocab(self):
        if self.lower:
            counter = Counter([line[self.idx].lower() for sent in self.data for line in sent])
        else:
            counter = Counter([line[self.idx] for sent in self.data for line in sent])
        for k in list(counter.keys()):
            if counter[k] < self.cutoff or k in self.ignore:
                del counter[k]

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}
        print(f"--vocab size:{len(self._unit2id)}")


class GraphVocab(BaseVocab):
    # eg:
    #   graphvocab = GraphVocab(data, self.args['shorthand'], idx=2)
    def build_vocab(self):
        deprels = []
        for sent in self.data:
            for line in sent:
                arcs = line[self.idx]
                arcs = arcs.split('|')
                for arc in arcs:
                    deprel = arc.split(':')[1]
                    deprels.append(deprel)
        counter = Counter(deprels)
        for k in list(counter.keys()):
            if counter[k] < self.cutoff:
                del counter[k]
        self._id_list = list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._id2unit = ['<EMPTY>', '<UNK>'] + self._id_list
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}
        # pprint(counter)
        # print('----------------')
        # print(len(counter))
        # pprint(len(self._id2unit))
        # print('------------------')

    def get_arc(self, sent, idx):
        res = []
        for w in sent:
            _res = []
            arcs = w[idx]
            arcs = arcs.split('|')
            for arc in arcs:
                head = arc.split(':')[0]
                deprel = arc.split(':')[1]
                deprel_idx = self.unit2id(deprel)
                _res.append([int(head), deprel_idx])
            res.append(_res)
        return res

    def parse_to_sent(self, inputs):
        sent = []
        for w in inputs:
            arc = []
            for a in w:
                head = str(a[0])
                deprel = self.id2unit(a[1])
                arc.append([head, deprel])
            if len(arc) == 1:
                string = ':'.join(arc[0])
                sent.append(string)
            else:
                string = ''
                for item in arc:
                    string += ':'.join(item) + '|'
                sent.append(string[:-1])
        return sent

    def parse_to_sent_batch(self, inputs):
        sents = []
        for s in inputs:
            words = []
            for w in s:
                arc = []
                for a in w:
                    head = str(a[0])
                    deprel = self.id2unit(a[1])
                    arc.append([head, deprel])
                if len(arc) == 1:
                    string = ':'.join(arc[0])
                    words.append(string)
                else:
                    string = ''
                    for item in arc:
                        string += ':'.join(item) + '|'
                    words.append(string[:-1])
            sents.append(words)
        return sents


class MultiVocab(BaseMultiVocab):
    # eg:
    #         vocab = MultiVocab({'char': charvocab,
    #                             'word': wordvocab,
    #                             'upos': uposvocab,
    #                             'graph': graphvocab})
    def state_dict(self):
        """ Also save a vocab name to class name mapping in state dict. """
        state = OrderedDict()
        key2class = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
            key2class[k] = type(v).__name__
        state['_key2class'] = key2class
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        class_dict = {
            'CharVocab': CharVocab,
            'WordVocab': WordVocab,
            'GraphVocab': GraphVocab
        }
        new = cls()
        assert '_key2class' in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict.pop('_key2class')
        for k, v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new
