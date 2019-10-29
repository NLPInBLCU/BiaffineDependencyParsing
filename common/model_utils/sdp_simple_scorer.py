"""
Utils and wrappers for scoring parsers.
"""
import sys

'''
def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for UD parser scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['LAS']
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['LAS', 'MLAS', 'BLEX']]
        print("LAS\tMLAS\tBLEX")
        print("{:.2f}\t{:.2f}\t{:.2f}".format(*scores))
    return p, r, f
'''

INF = float('inf')


def conllu_file_2_sem16_file(conllu_filename):
    """
    将conllu格式转化为sem16的sdp格式，然后利用新的score评价指标评价
    :param conllu_filename:
    :return:
    """

    with open(conllu_filename, encoding='utf-8') as f:
        with open(conllu_filename + '.sem16.sdp', 'w', encoding='utf-8') as g:
            buff = []
            for line in f:
                line = line.strip('\n')
                items = line.split('\t')
                if len(items) == 10:
                    # Add it to the buffer
                    buff.append(items)
                elif buff:
                    # g.write('#\n')
                    # Process the buffer
                    for i, items in enumerate(buff):
                        # words.append(line)
                        if items[8] != '_':
                            nodes = items[8].split('|')
                            for node in nodes:
                                words = items
                                # copy xpos to upos
                                words[3] = words[4]
                                node = node.split(':', 1)
                                node[0] = int(node[0])
                                words[6], words[7], words[8] = str(node[0]), node[1], '_'
                                # words[7] = node[1]
                                # words[8] = '_'
                                g.write('\t'.join(words) + '\n')
                        else:
                            g.write('\t'.join(items) + '\n')
                    g.write('\n')
                    buff = []
    return conllu_filename + '.sem16.sdp'


def stat_one_tree(lines):
    stat_data = {}
    for line in lines:
        payload = line.strip().split("\t")
        if (len(payload) < 7):
            print(lines)
        id_val = int(payload[0])
        form_val = payload[1]
        postag_val = payload[3]
        head_val = payload[6]
        deprel_val = payload[7]
        # if not opts.punctuation and engine(form_val, postag_val):
        #     continue
        if id_val not in stat_data:
            stat_data[id_val] = {
                "id": id_val,
                "form": form_val,
                "heads": [head_val],
                "deprels": [deprel_val]
            }
        else:
            assert (form_val == stat_data[id_val]["form"])
            stat_data[id_val]["heads"].append(head_val)
            stat_data[id_val]['deprels'].append(deprel_val)
    return stat_data


def stat_one_node_heads_and_deprels(gold_heads, gold_deprels, test_heads, test_deprels):
    gold_len = len(gold_heads)  # ! assert( len(gold_heads) == len(gold_deprels))
    test_len = len(test_heads)
    nr_right_heads = 0
    nr_right_deprels = 0

    assert gold_len != 0 and test_len != 0
    if gold_len == 1 and test_len == 1:
        # ! normal situation
        if gold_heads[0] == test_heads[0]:
            nr_right_heads = 1
            if gold_deprels[0] == test_deprels[0]:
                nr_right_deprels = 1
    else:
        for gold_head, gold_deprel in zip(gold_heads, gold_deprels):
            if gold_head in test_heads:
                nr_right_heads += 1
                head_idx = test_heads.index(gold_head)
                if gold_deprel == test_deprels[head_idx]:  # !! head_idx == deprel_idx
                    nr_right_deprels += 1
    return (gold_len, test_len,
            nr_right_heads, nr_right_deprels)


def stat_gold_and_test_data(gold_stat_data, test_stat_data):
    nr_gold_rels = 0
    nr_test_rels = 0
    nr_head_right = 0
    nr_deprel_right = 0

    for idx in gold_stat_data.keys():
        gold_node = gold_stat_data[idx]
        test_node = test_stat_data[idx]
        assert (gold_node['id'] == test_node['id'])

        (
            gold_rels_len, test_rels_len,
            nr_one_node_right_head, nr_one_node_right_deprel
        ) = (
            stat_one_node_heads_and_deprels(gold_node['heads'], gold_node['deprels'],
                                            test_node['heads'], test_node['deprels'])
        )

        nr_gold_rels += gold_rels_len
        nr_test_rels += test_rels_len
        nr_head_right += nr_one_node_right_head
        nr_deprel_right += nr_one_node_right_deprel

    return (nr_gold_rels, nr_test_rels,
            nr_head_right, nr_deprel_right)


def score(system_conllu_file, gold_conllu_file):
    system_sem16_file = conllu_file_2_sem16_file(system_conllu_file)
    gold_sem16_file = conllu_file_2_sem16_file(gold_conllu_file)
    reference_dataset = open(gold_sem16_file, "r", encoding='utf-8').read().strip().split("\n\n")
    answer_dataset = open(system_sem16_file, "r", encoding='utf-8').read().strip().split("\n\n")

    assert len(reference_dataset) == len(answer_dataset), "Number of instance unequal."
    nr_total_gold_rels = 0
    nr_total_test_rels = 0
    nr_total_right_heads = 0
    nr_total_right_deprels = 0

    nr_sentence = len(reference_dataset)

    length_error_num = 0

    for reference_data, answer_data in zip(reference_dataset, answer_dataset):
        reference_lines = reference_data.split("\n")
        answer_lines = answer_data.split("\n")

        reference_stat_data = stat_one_tree(reference_lines)
        answer_stat_data = stat_one_tree(answer_lines)
        if len(reference_stat_data) != len(answer_stat_data):
            length_error_num += 1
            continue

        (
            nr_one_gold_rels, nr_one_test_rels,
            nr_one_head_right, nr_one_deprel_right
        ) \
            = stat_gold_and_test_data(reference_stat_data, answer_stat_data)

        nr_total_gold_rels += nr_one_gold_rels
        nr_total_test_rels += nr_one_test_rels
        nr_total_right_heads += nr_one_head_right
        nr_total_right_deprels += nr_one_deprel_right

    nr_sentence -= length_error_num

    LAS = float(2 * nr_total_right_deprels) / (nr_total_test_rels + nr_total_gold_rels) \
        if (nr_total_gold_rels + nr_total_test_rels) != 0 else INF

    UAS = float(2 * nr_total_right_heads) / (nr_total_test_rels + nr_total_gold_rels) \
        if (nr_total_gold_rels + nr_total_test_rels) != 0 else INF
    return UAS, LAS


def parse_conllu(f_object):
    sents = []
    sent = []
    for line in f_object:
        line = line.strip()
        if len(line) == 0:
            sents.append(sent)
            sent = []
        else:
            line = line.split('\t')
            sent.append(line[8])
    return sents


def old_score(system_conllu_file, gold_conllu_file):
    arc_total, arc_correct, arc_predict, label_total, label_correct, label_predict = 0, 0, 0, 0, 0, 0
    with open(system_conllu_file, 'r', encoding='utf-8') as f_system, open(gold_conllu_file, 'r',
                                                                           encoding='utf-8') as f_gold:

        sys_sents = []
        sys_sent = []
        for line in f_system:
            line = line.strip()
            if len(line) == 0:
                sys_sents.append(sys_sent)
                sys_sent = []
            else:
                line = line.split('\t')
                sys_sent.append(line[8])

        gold_sents = []
        gold_sent = []
        for line in f_gold:
            line = line.strip()
            if len(line) == 0:
                gold_sents.append(gold_sent)
                gold_sent = []
            else:
                line = line.split('\t')
                gold_sent.append(line[8])

        for sys_sent, gold_sent in zip(sys_sents, gold_sents):
            for system, gold in zip(sys_sent, gold_sent):
                gold = gold.split('|')
                system = system.split('|')

                label_total += len(gold)
                label_predict += len(system)
                label_correct += len(list(set(gold) & set(system)))

                gold_head = [arc.split(':')[0] for arc in gold]
                sys_head = [arc.split(':')[0] for arc in system]

                arc_total += len(gold_head)
                arc_predict += len(sys_head)
                arc_correct += len(list(set(gold_head) & set(sys_head)))

    arc_recall = arc_correct / arc_total
    arc_precison = arc_correct / arc_predict
    arc_f = 2 * arc_precison * arc_recall / (arc_precison + arc_recall)

    label_recall = label_correct / label_total
    label_precison = label_correct / label_predict
    label_f = 2 * label_precison * label_precison / (label_precison + label_recall)
    UAS = arc_f
    LAS = label_f
    # print('UAS Score:{}'.format(UAS))
    # print('LAS Score:{}'.format(LAS))
    return UAS, LAS


if __name__ == '__main__':
    pass
    # from run_utils.timer import Timer
    #
    # system_conllu_file = './sdp_text_test_predict2.conllu'
    # gold_conllu_file = './sdp_text_test.conllu'
    # with Timer('new'):
    #     for _ in range(1):
    #         UAS, LAS = score(system_conllu_file, gold_conllu_file)
    # with Timer('old'):
    #     for _ in range(1):
    #         old_UAS, old_LAS = old_score(system_conllu_file, gold_conllu_file)
    # print(f'new LAS:{LAS:.5f}\tnew UAS:{UAS:.5f}')
    # print(f'old LAS:{old_LAS:.5f}\told UAS:{old_UAS:.5f}')
