# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24
"""
    在train/test/inference之前必须先建立依存标签的vocab,
    在train/test/inference时应该使用同一份vocab
"""
from pathlib import Path
from collections import Counter
from utils.input_utils.conll_file import load_conllu_file


def build_vocab(data_dir, cutoff=1):
    deprels = []
    data_dir = Path(data_dir)
    file_data = []
    for conllu_file in data_dir.glob("*/*.conllu"):
        print(f'Loading {str(conllu_file)} ...')
        _, _data = load_conllu_file(str(conllu_file))
        file_data += _data.sentences
    max_char_length = 0
    max_word_length = 0
    max_char_length_sent = ''
    max_word_length_sent = ''
    for sent in file_data:
        words = []
        for line in sent.words:
            arcs = line.dep
            words.append(line.word)
            arcs = arcs.split('|')
            for arc in arcs:
                deprel = arc.split(':')[1]
                deprels.append(deprel)
        if len(words) > max_word_length:
            max_word_length = len(words)
            max_word_length_sent = ''.join(words)
        if len(''.join(words)) > max_char_length:
            max_char_length = len(''.join(words))
            max_char_length_sent = ''.join(words)
    counter = Counter(deprels)
    for k in list(counter.keys()):
        if counter[k] < cutoff:
            del counter[k]
    id_list = list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
    id2unit = ['<EMPTY>', '<UNK>'] + id_list
    with open('graph_vocab.txt', 'w', encoding='utf-8')as f:
        for u in id2unit:
            f.write(u + '\n')
    print(f'max char length sent:')
    print(max_char_length_sent)
    print(f'max char length : {max_char_length}')
    print(f'max word length sent:')
    print(max_word_length_sent)
    print(f'max word length : {max_word_length}')
    return counter


if __name__ == '__main__':
    from pprint import pprint

    pprint(build_vocab('.'))
