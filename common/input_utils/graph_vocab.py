# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24


class GraphVocab(object):
    def __init__(self, vocab_file):
        self.id2unit = ['<EMPTY>', '<UNK>']
        with open(vocab_file, encoding='utf-8')as f:
            for line in f:
                if line.strip() not in self.id2unit:
                    self.id2unit.append(line.strip())
        self.unit2id = {u: i for i, u in enumerate(self.id2unit)}

    def get_labels(self):
        return self.id2unit

    def parse_to_sent_batch(self, inputs):
        sents = []
        for s in inputs:
            words = []
            for w in s:
                arc = []
                for a in w:
                    head = str(a[0])
                    deprel = self.id2unit[a[1]]
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


if __name__ == '__main__':
    vocab = GraphVocab('../dataset/graph_vocab.txt')
    print(vocab.id2unit)
    print(vocab.unit2id)
