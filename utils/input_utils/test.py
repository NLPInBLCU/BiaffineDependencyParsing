# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24
from utils.input_utils.conll_file import load_conllu_file
from pathlib import Path
from NLP_utils.easy_ltp import EasyLTP

if __name__ == '__main__':

    # 测试读取：
    ltp = EasyLTP('/home/liangs/disk/data/ltp_data_v3.4.0')
    conllu_path = Path('/home/liangs/codes/trivial_codes/wanglihui')
    all_sents = []
    for file in conllu_path.glob('*.conllu'):
        # file_name = "../dataset/test/sdp_text_test.conllu"
        conllu_file, data = load_conllu_file(str(file))
        # data = data[:1]
        for sent in data:
            # for w in sent:
            #     print(w)
            # print()
            words = []
            for w in sent:
                words.append(w[0])
            all_sents.append(words)
    with open('wanglihui.txt', 'w', encoding='utf-8')as f:
        for sent in all_sents:
            pos = ltp.pos_words(sent)
            f.write('\t'.join(sent) + '\n')
            f.write('\t'.join(pos) + '\n')
            f.write('\n')
    # print(all_sents[:3])
    # vocab = GraphVocab(data, idx=2)
    # for sent in data:
    #     for s, r in zip(sent, vocab.get_arc(sent, 2)):
    #         print(s, r)
