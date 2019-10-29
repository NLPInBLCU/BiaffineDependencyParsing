# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from common.input_utils.conll_file import load_conllu_file
from common.input_utils.graph_vocab import GraphVocab
from pytorch_transformers import BertTokenizer, RobertaTokenizer, XLMTokenizer, XLNetTokenizer

BERT_TOKENIZER = {
    'bert': BertTokenizer,
    'xlnet': XLNetTokenizer,
    'xlm': XLMTokenizer,
    'roberta': RobertaTokenizer,
}


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, sentence, start_pos, end_pos, deps=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.sentence = sentence
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.deps = deps


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, dep_ids, start_pos=None, end_pos=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.dep_ids = dep_ids
        self.start_pos = start_pos
        self.end_pos = end_pos


class CoNLLUProcessor(object):
    """
        依存分析BERT数据处理，输入文件必须是CoNLL-U格式
    """

    def __init__(self, args, graph_vocab, word_vocab=None):
        assert isinstance(graph_vocab, GraphVocab)
        self.graph_vocab = graph_vocab
        self.args = args
        assert args.encoder_type == 'bert', "暂时不支持xlent等类似BERT的模型，输入端需要适配（ROOT,start_pos,end_pos等等）"
        # TODO：支持xlnet,roberta,xlm等模型
        self.word_vocab = word_vocab

    # def get_train_examples(self, data_dir, file_name, max_seq_length):
    #     """Gets a collection of `InputExample`s for the train set."""
    #     CoNLLU_file, CoNLLU_data = load_conllu_file(os.path.join(data_dir, file_name))
    #     return self._create_bert_example(CoNLLU_data, 'train', max_seq_length), CoNLLU_file

    def get_examples(self, data_dir, file_name, max_seq_length):
        """Gets a collection of `InputExample`s for the dev set."""
        CoNLLU_file, CoNLLU_data = load_conllu_file(os.path.join(data_dir, file_name))
        return self._create_bert_example(CoNLLU_data, 'dev', max_seq_length), CoNLLU_file

    def _get_words_start_end_pos(self, words_list, max_seq_length):
        s = []
        e = []
        # 0 for ROOT if root_representation == [CLS]
        if self.args.root_representation == 'cls':
            s.append(0)
            e.append(0)
        else:
            # 1 for other root_representation
            s.append(1)
            e.append(1)
        if self.args.root_representation != 'cls':
            # 如果ROOT的表示不是CLS，那么words_list第一个元素就是ROOT，应当跳过（因为前面已经用0或者1表示了）
            clear_words_list = words_list[1:]
        else:
            clear_words_list = words_list
        #  BERT For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        # 如果ROOT是用[CLS]表示，则从1开始计算；否则，则从2开始计算
        # 0永远是CLS
        if self.args.root_representation == 'cls':
            sum_len = 1
        else:
            sum_len = 2
        for w in clear_words_list:
            s.append(sum_len) if sum_len < max_seq_length else s.append(max_seq_length - 1)
            if w.lower() not in self.word_vocab:  # 如果单词不在词表中，说明会被分割，要分开计算
                sum_len += len(w)
            else:  # 如果单词在词表中，可以视长度为1
                sum_len += 1
            e.append(sum_len - 1) if sum_len - 1 < max_seq_length else e.append(max_seq_length - 1)
        return s, e

    def _create_bert_example(self, CoNLLU_data, set_type, max_seq_length):
        examples = []
        # print(CoNLLU_data)
        for i, sent in enumerate(CoNLLU_data):
            guid = f"{set_type}-{i}"
            words = []
            deps = []
            if self.args.root_representation == 'cls':
                # BERT输入会自动添加一个CLS，所以此时不需要特殊处理
                pass
            elif self.args.root_representation == 'unused':
                # BERT中预留了部分未经训练的空位，这里使用unused1 （1~99都可以）
                words.append('[unused1]')
                pass
            elif self.args.root_representation in ['root', '根']:
                # 其他可能的表示形式，注意，这些字符已经被BERT预训练过，所以可能效果不佳
                words.append(self.args.root_representation)
            else:
                raise Exception(f'illegal root representation:{self.args.root_representation}')
            for line in sent:
                line_res = []
                if line[-1] == '_':
                    if deps:
                        # print(CoNLLU_data)
                        raise Exception('illegal CoNLLU data')
                    words.append(line[0])
                    continue
                arcs = line[-1].split('|')
                for arc in arcs:
                    head, dep_rel = arc.split(':')
                    dep_rel_idx = self.graph_vocab.unit2id[dep_rel]
                    line_res.append([int(head), dep_rel_idx])
                deps.append(line_res)
                words.append(line[0])
            if not deps:
                deps = None
            start_pos, end_pos = self._get_words_start_end_pos(words, max_seq_length)
            # print(words)
            examples.append(
                InputExample(guid=guid, sentence="".join(words), start_pos=start_pos, end_pos=end_pos, deps=deps)
            )
        return examples


def _make_label_target(arcs, max_seq_length):
    if arcs:
        graphs = [[0] * max_seq_length for _ in range(max_seq_length)]
        for word_idx, word in enumerate(arcs, start=1):
            for arc in word:
                head_idx = arc[0]
                rel_idx = arc[1]
                graphs[word_idx][head_idx] = rel_idx
    else:
        graphs = [[-1] * max_seq_length for _ in range(max_seq_length)]
    return graphs


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 skip_too_long_input=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    assert not cls_token_at_end, "CLS必须在句首，目前不支持xlnet"

    features = []
    skip_input_num = 0
    for (ex_index, example) in enumerate(examples):
        assert isinstance(example, InputExample)
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # print(example.sentence)
        tokens_a = tokenizer.tokenize(example.sentence)
        # print('tokens:')
        # print(tokens_a)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        # 为ROOT表示等预留足够的空间
        special_tokens_count += 3
        if len(tokens_a) > max_seq_length - special_tokens_count:
            if skip_too_long_input:
                # 这里直接跳过过长的句子
                skip_input_num += 1
                continue
            else:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        start_pos = example.start_pos
        end_pos = example.end_pos
        # print(end_pos)

        if start_pos:
            assert len(start_pos) == len(end_pos)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        pos_padding_length = max_seq_length - len(start_pos)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            start_pos = ([max_seq_length - 1] * pos_padding_length) + start_pos
            end_pos = ([max_seq_length - 1] * pos_padding_length) + end_pos
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            start_pos = start_pos + ([max_seq_length - 1] * pos_padding_length)
            end_pos = end_pos + ([max_seq_length - 1] * pos_padding_length)
        # print(end_pos)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_pos) == len(end_pos) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        dep_ids = _make_label_target(example.deps, max_seq_length)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          dep_ids=dep_ids,
                          start_pos=start_pos,
                          end_pos=end_pos))
    if skip_input_num > 0:
        print(f'\n>> convert_examples_to_features skip input:{skip_input_num} !!')
    return features


def load_and_cache_examples(args, graph_vocab, tokenizer, train=True, dev=True, test=False):
    word_vocab = tokenizer.vocab if args.encoder_type == 'bert' else None
    processor = CoNLLUProcessor(args, graph_vocab, word_vocab)
    # Load data features from cache or dataset file
    # cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
    #     'dev' if evaluate else 'train',
    #     list(filter(None, args.model_path.split('/'))).pop(),
    #     str(args.max_seq_length)))
    # if os.path.exists(cached_features_file):
    #     # logger.info("Loading features from cached file %s", cached_features_file)
    #     features = torch.load(cached_features_file)
    # else:
    # logger.info("Creating features from dataset file at %s", args.data_dir)

    label_list = graph_vocab.get_labels()
    if test:
        test_examples, test_CoNLLU_file = processor.get_examples(args.data_dir, args.test_file, args.max_seq_len)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_len, tokenizer,
                                                     cls_token_at_end=bool(args.encoder_type in ['xlnet']),
                                                     # xlnet has a cls token at the end
                                                     cls_token=tokenizer.cls_token,
                                                     cls_token_segment_id=2 if args.encoder_type in ['xlnet'] else 0,
                                                     sep_token=tokenizer.sep_token,
                                                     sep_token_extra=bool(args.encoder_type in ['roberta']),
                                                     # roberta uses an extra separator b/w pairs of sentences,
                                                     # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                     pad_on_left=bool(args.encoder_type in ['xlnet']),
                                                     # pad on the left for xlnet
                                                     pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                         0],
                                                     pad_token_segment_id=4 if args.encoder_type in ['xlnet'] else 0,
                                                     skip_too_long_input=args.skip_too_long_input,
                                                     )
        # print("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        test_dataset = make_dataset(args, test_features)
    else:
        test_dataset = test_CoNLLU_file = None
    if dev:
        dev_examples, dev_CoNLLU_file = processor.get_examples(args.data_dir, args.dev_file, args.max_seq_len)
        dev_features = convert_examples_to_features(dev_examples, label_list, args.max_seq_len, tokenizer,
                                                    cls_token_at_end=bool(args.encoder_type in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.encoder_type in ['xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.encoder_type in ['roberta']),
                                                    # roberta uses an extra separator b/w pairs of sentences,
                                                    # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(args.encoder_type in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.encoder_type in ['xlnet'] else 0,
                                                    skip_too_long_input=args.skip_too_long_input,
                                                    )
        # print("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)

        ## Convert to Tensors and build dataset
        dev_dataset = make_dataset(args, dev_features)
    else:
        dev_dataset = dev_CoNLLU_file = None
    if train:
        train_examples, train_CoNLLU_file = processor.get_examples(args.data_dir, args.train_file, args.max_seq_len)
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_len, tokenizer,
                                                      cls_token_at_end=bool(args.encoder_type in ['xlnet']),
                                                      # xlnet has a cls token at the end
                                                      cls_token=tokenizer.cls_token,
                                                      cls_token_segment_id=2 if args.encoder_type in ['xlnet'] else 0,
                                                      sep_token=tokenizer.sep_token,
                                                      sep_token_extra=bool(args.encoder_type in ['roberta']),
                                                      # roberta uses an extra separator b/w pairs of sentences,
                                                      # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                      pad_on_left=bool(args.encoder_type in ['xlnet']),
                                                      # pad on the left for xlnet
                                                      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                          0],
                                                      pad_token_segment_id=4 if args.encoder_type in ['xlnet'] else 0,
                                                      skip_too_long_input=args.skip_too_long_input,
                                                      )
        # print("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)

        ## Convert to Tensors and build dataset
        train_dataset = make_dataset(args, train_features)
    else:
        train_dataset = train_CoNLLU_file = None
    return train_dataset, train_CoNLLU_file, dev_dataset, dev_CoNLLU_file, test_dataset, test_CoNLLU_file


def make_dataset(args, features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_pos = torch.tensor([t.start_pos for t in features], dtype=torch.long)
    # print([t.end_pos for t in features])
    all_end_pos = torch.tensor([t.end_pos for t in features], dtype=torch.long)
    all_dep_ids = torch.tensor([t.dep_ids for t in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_pos, all_end_pos, all_dep_ids)
    return dataset


def get_data_loader(dataset, batch_size, evaluation=False):
    if evaluation:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return data_loader


def load_bert_tokenizer(model_path, model_type, do_lower_case=True):
    # 必须把 unused 添加到 additional_special_tokens 上，否则 unused （用来表示ROOT）可能无法正确切分！
    return BERT_TOKENIZER[model_type].from_pretrained(model_path, do_lower_case=do_lower_case,
                                                      additional_special_tokens=['[unused1]', '[unused2]', '[unused3]'])


def load_input(args, train=True, dev=True, test=False):
    tokenizer = load_bert_tokenizer(args.bert_path, args.encoder_type)
    vocab = GraphVocab(args.graph_vocab_file)
    train_dataset, train_conllu, dev_dataset, dev_conllu, test_dataset, test_conllu = \
        load_and_cache_examples(args, vocab, tokenizer, train=train, dev=dev, test=test)
    if train_dataset:
        train_data_loader = get_data_loader(train_dataset, batch_size=args.train_batch_size, evaluation=False)
    else:
        train_data_loader = None
    if dev_dataset:
        dev_data_loader = get_data_loader(dev_dataset, batch_size=args.eval_batch_size, evaluation=True)
    else:
        dev_data_loader = None
    if test_dataset:
        test_data_loader = get_data_loader(test_dataset, batch_size=args.eval_batch_size, evaluation=True)
    else:
        test_data_loader = None
    return train_data_loader, train_conllu, dev_data_loader, dev_conllu, test_data_loader, test_conllu


if __name__ == '__main__':
    class Args():
        def __init__(self):
            self.model_path = '/home/liangs/disk/data/bert-base-chinese'
            self.data_dir = '/home/liangs/codes/doing_codes/CSDP_Biaffine_Parser_lhy/CSDP_Biaffine_Parser_lhy/dataset'
            self.train_file = 'test.conllu'
            self.max_seq_len = 10
            self.encoder_type = 'bert'
            self.root_representation = 'unused'
            self.device = 'cpu'


    args = Args()
    # print(f'{}')
    tokenizer = load_bert_tokenizer('/home/liangs/disk/data/bert-base-chinese', 'bert')
    # print(tokenizer)
    # print(tokenizer.vocab['[unused1]'])
    vocab = GraphVocab('/home/liangs/codes/doing_codes/CSDP_Biaffine_Parser/dataset/graph_vocab.txt')
    dataset, CoNLLU_file, _, _, _, _ = load_and_cache_examples(args, vocab, tokenizer, train=True, dev=False,
                                                               test=False)
    data_loader = get_data_loader(dataset, 3, evaluation=True)
    # 原始输入句子
    print(CoNLLU_file.get(['word'], as_sentences=True))
    for batch in data_loader:
        # input ids:
        print(batch[0])
        # input mask:
        print(batch[1])
        # start pos:
        print(batch[3])
        # labeled target
        print(batch[-1])
        # unlabeled target
        # print(batch[-1].ge(1).to(batch[-1].dtype))
