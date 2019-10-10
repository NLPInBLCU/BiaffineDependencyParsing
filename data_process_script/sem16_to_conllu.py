# -*- coding: utf-8 -*-

def to_conllu(sdp_filename, conllu_filename):
    with open(sdp_filename, encoding='utf-8') as f, open(conllu_filename, 'w', encoding='utf-8') as g:
        sents = f.read().strip().split('\n\n')
        for sent in sents:
            conllu_form = []
            words = []
            lines = sent.strip().split('\n')
            for line in lines:
                if line.startswith('#'):
                    conllu_form.append(line)
                    continue
                items = line.strip().split('\t')
                print(items)
                if int(items[0]) == len(words) + 1:
                    if items[6] is not '_' and items[7] is not '_':
                        items[8] = items[6] + ':' + items[7]
                    words.append(items)
                elif int(items[0]) == len(words):
                    words[-1][8] += '|' + items[6] + ':' + items[7]
                else:
                    print("Error:{}".format(line))
            for word in words:
                conllu_form.append('\t'.join(word))
            g.write('\n'.join(conllu_form) + '\n\n')


# ***************************************************************
if __name__ == '__main__':
    sdp_filename = './text.valid.conll'
    output_filename = './sdp_text_dev.conllu'
    to_conllu(sdp_filename, output_filename)
