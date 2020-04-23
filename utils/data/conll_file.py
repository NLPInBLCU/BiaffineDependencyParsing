"""
A wrapper/loader for the official conll-u format files.
"""
import os
import io
from typing import List, Tuple

FIELD_NUM = 10

FIELD_TO_IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4,
                'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8,
                'misc': 9}


class CoNLLFile(object):
    def __init__(
            self,
            filename=None,
            input_str=None,
            ignore_gapping=True
    ):
        # If ignore_gapping is True, all words that are gap fillers
        # (identified with a period in the sentence index) will be ignored.
        self.ignore_gapping = ignore_gapping
        if filename is not None and not os.path.exists(filename):
            raise Exception("File not found at: " + filename)
        if filename:
            assert filename.endswith('conllu'), "Loaded file must be conllu file."
        if filename is None:
            assert input_str is not None and len(input_str) > 0
            self._file = input_str
            self._from_str = True
        else:
            self._file = filename
            self._from_str = False

    def load_all(self):
        """ Trigger all lazy initializations so that the file is loaded."""
        _ = self.sents
        _ = self.num_words

    def load_conll(self):
        """
        Load data into a list of sentences, where each sentence is a list of lines,
        and each line is a list of conllu fields.
        """
        sents, cache = [], []
        if self._from_str:
            infile = io.StringIO(self.file)
        else:
            infile = open(self.file, encoding='utf-8')
        while True:
            line = infile.readline()
            if len(line) == 0:
                break
            line = line.strip()
            if len(line) == 0:
                if len(cache) > 0:
                    sents.append(cache)
                    cache = []
            else:
                if line.startswith('#'):  # skip comment line
                    continue
                array = line.split('\t')
                if self.ignore_gapping and '.' in array[0]:
                    continue
                assert len(array) == FIELD_NUM
                cache += [array]
        if len(cache) > 0:
            sents.append(cache)
        if not self._from_str:
            infile.close()
        return sents

    @property
    def file(self):
        return self._file

    @property
    def sents(self):
        if not hasattr(self, '_sents'):
            self._sents = self.load_conll()
        return self._sents

    def __len__(self):
        return len(self.sents)

    @property
    def num_words(self):
        """ Num of total words, after multi-word expansion."""
        if not hasattr(self, '_num_words'):
            n = 0
            for sent in self.sents:
                for ln in sent:
                    if '-' not in ln[0]:
                        n += 1
            self._num_words = n
        return self._num_words

    def get(self, fields, as_sentences=False):
        """ Get fields from a list of field names. If only one field name is provided, return a list
        of that field; if more than one, return a list of list. Note that all returned fields are after
        multi-word expansion.
        """
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."
        field_idxs = [FIELD_TO_IDX[f.lower()] for f in fields]
        results = []
        for sent in self.sents:
            cursent = []
            for ln in sent:
                if '-' in ln[0]:  # skip
                    continue
                if len(field_idxs) == 1:
                    cursent += [ln[field_idxs[0]]]
                else:
                    cursent += [[ln[fid] for fid in field_idxs]]

            if as_sentences:
                results.append(cursent)
            else:
                results += cursent
        return results

    def set(self, fields, contents):
        """
        Set fields based on contents. If only one field (singleton list) is provided,
        then a list of content will be expected; otherwise a list of list of contents will be expected.
        """
        assert isinstance(fields, list), "Must provide field names as a list."
        assert isinstance(contents, list), "Must provide contents as a list (one item per line)."
        assert len(fields) >= 1, "Must have at least one field."
        if self.num_words != len(contents):
            print("Contents must have the same number as the original file.")
        field_idxs = [FIELD_TO_IDX[f.lower()] for f in fields]
        cidx = 0
        for sent in self.sents:
            for ln in sent:
                if '-' in ln[0]:
                    continue
                if len(field_idxs) == 1:
                    ln[field_idxs[0]] = contents[cidx]
                    if fields == ['deps']:
                        head_value, deprel_value = contents[cidx].split('|')[0].split(':')
                        ln[FIELD_TO_IDX['head']] = head_value
                        ln[FIELD_TO_IDX['deprel']] = deprel_value
                else:
                    for fid, ct in zip(field_idxs, contents[cidx]):
                        ln[fid] = ct
                cidx += 1
        return

    def write_conll(self, filename):
        """ Write current conll contents to file.
        """
        conll_string = self.conll_as_string()
        with open(filename, 'w', encoding='utf-8') as outfile:
            outfile.write(conll_string)
        return

    def conll_as_string(self):
        """ Return current conll contents as string
        """
        return_string = ""
        for sent in self.sents:
            for ln in sent:
                return_string += ("\t".join(ln) + "\n")
            return_string += "\n"
        return return_string

    def write_conll_with_lemmas(self, lemmas, filename):
        """ Write a new conll file, but use the new lemmas to replace the old ones."""
        assert self.num_words == len(lemmas), "Num of lemmas does not match the number in original data file."
        lemma_idx = FIELD_TO_IDX['lemma']
        idx = 0
        with open(filename, 'w') as outfile:
            for sent in self.sents:
                for ln in sent:
                    if '-' not in ln[0]:  # do not process if it is a mwt line
                        lm = lemmas[idx]
                        if len(lm) == 0:
                            lm = '_'
                        ln[lemma_idx] = lm
                        idx += 1
                    print("\t".join(ln), file=outfile)
                print("", file=outfile)
        return

    def get_mwt_expansions(self):
        word_idx = FIELD_TO_IDX['word']
        expansions = []
        src = ''
        dst = []
        for sent in self.sents:
            mwt_begin = 0
            mwt_end = -1
            for ln in sent:
                if '.' in ln[0]:
                    # skip ellipsis
                    continue

                if '-' in ln[0]:
                    mwt_begin, mwt_end = [int(x) for x in ln[0].split('-')]
                    src = ln[word_idx]
                    continue

                if mwt_begin <= int(ln[0]) < mwt_end:
                    dst += [ln[word_idx]]
                elif int(ln[0]) == mwt_end:
                    dst += [ln[word_idx]]
                    expansions += [[src, ' '.join(dst)]]
                    src = ''
                    dst = []

        return expansions

    def get_mwt_expansion_cands(self):
        word_idx = FIELD_TO_IDX['word']
        cands = []
        for sent in self.sents:
            for ln in sent:
                if "MWT=Yes" in ln[-1]:
                    cands += [ln[word_idx]]

        return cands

    def write_conll_with_mwt_expansions(self, expansions, output_file):
        """ Expands MWTs predicted by the tokenizer and write to file.
        This method replaces the head column with a right branching tree. """
        idx = 0
        count = 0

        for sent in self.sents:
            for ln in sent:
                idx += 1
                if "MWT=Yes" not in ln[-1]:
                    print("{}\t{}".format(idx, "\t".join(ln[1:6] + [str(idx - 1)] + ln[7:])), file=output_file)
                else:
                    # print MWT expansion
                    expanded = [x for x in expansions[count].split(' ') if len(x) > 0]
                    count += 1
                    endidx = idx + len(expanded) - 1

                    print("{}-{}\t{}".format(idx, endidx,
                                             "\t".join(['_' if i == 5 or i == 8 else x for i, x in enumerate(ln[1:])])),
                          file=output_file)
                    for e_i, e_word in enumerate(expanded):
                        print("{}\t{}\t{}".format(idx + e_i, e_word,
                                                  "\t".join(['_'] * 4 + [str(idx + e_i - 1)] + ['_'] * 3)),
                              file=output_file)
                    idx = endidx

            print("", file=output_file)
            idx = 0

        assert count == len(expansions), "{} {} {}".format(count, len(expansions), expansions)
        return


class CoNLLUWord(object):
    def __init__(self, word_data):
        self.word = word_data[0]
        self.pos = word_data[1]
        self.dep = word_data[2]


class CoNLLUSent(object):
    def __init__(self, conllu_sent: List):
        self.words = []
        for word in conllu_sent:
            self.words.append(CoNLLUWord(word))


class CoNLLUData(object):
    def __init__(self, conllu_data: List):
        self.sentences = []
        for sent in conllu_data:
            self.sentences.append(CoNLLUSent(sent))


def load_conllu_file(filename):
    """
    Conllu file 格式:
        单词序号\t单词\t单词\t词性\t词性\t_(置空)\t父节点序号\t依存弧标签\tdeps(父节点序号:依存弧标签|...)
    :param filename:
    :param evaluation:
    :return:
    """
    conll_file = CoNLLFile(filename)
    data = conll_file.get(['word', 'upos', 'deps'], as_sentences=True)
    # data= [
    #             [ #sent1
    #                   [word1,pos1,deps1],
    #                   [word2,pos2,deps2],
    #             ],
    #             [ #sent2
    #             ]
    #       ]
    # 示例输出（一句话）：
    # [['人类', 'NN', '2:Poss'],
    #   ['生活', 'NN', '4:Exp|6:Exp'],
    #   ['的', 'DEG', '2:mAux'],
    #   ['舒适', 'AN', '6:eCoo'],
    #   ['和', 'CC', '6:mConj'],
    #   ['方便', 'AN', '17:Cont'],
    #   ['，', 'PU', '6:mPunc'],
    #   ['是', 'AD', '17:mMod'],
    #   ['连', 'AD', '13:mMod'],
    #   ['过去', 'NT', '13:Time'],
    #   ['的', 'DEG', '10:mAux'],
    #   ['王公', 'NN', '13:eCoo'],
    #   ['贵族', 'NN', '17:Aft'],
    #   ['也', 'AD', '17:mMod'],
    #   ['不', 'AD', '16:mNeg'],
    #   ['敢', 'VV', '17:mMod'],
    #   ['想', 'VV', '0:Root'],
    #   ['的', 'SP', '17:mTone'],
    #   ['。', 'PU', '17:mPunc']]
    return conll_file, CoNLLUData(data)


if __name__ == '__main__':
    from pprint import pprint

    # 测试读取：
    file_name = "../dataset/test/sdp_text_test.conllu"
    conllu_file, data = load_conllu_file(file_name)
    # pprint(data[:3])
    # 测试写入：
