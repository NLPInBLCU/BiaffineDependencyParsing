# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24
import json
import os
import pathlib
import pickle
from collections import Counter
from multiprocessing import Pool
from functools import partial
import torch
from utils.data.bertology_base import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers import BertTokenizer, RobertaTokenizer, XLMTokenizer, XLNetTokenizer

from utils.data.conll_file import CoNLLUData
from utils.data.custom_dataset import ConcatTensorRandomDataset
from utils.logger import get_logger
from utils.timer import Timer

BERTology_TOKENIZER = {
    'bert': BertTokenizer,
    'xlnet': XLNetTokenizer,
    'xlm': XLMTokenizer,
    'roberta': RobertaTokenizer,
}


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
                                 # sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 # skip_too_long_input=True,
                                 pos_tokenizer=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    assert not cls_token_at_end, "CLS必须在句首，目前不支持xlnet"
    assert not pad_on_left, "PAD必须在句子右侧，目前不支持xlnet"
    features = []
    # skip_input_num = 0
    for example in examples:
        assert isinstance(example, InputExample)
        tokens_a = tokenizer.tokenize(example.sentence)
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        # 为ROOT表示等预留足够的空间(目前至少预留5个位置)
        special_tokens_count += 3
        if len(tokens_a) > max_seq_length - special_tokens_count:
            # 由于无论是跳过过长的句子还是截断过长的句子都会导致在dev时无法写入文件，
            # 这里我们暂时不允许过小的max_seq_len
            # todo: 基于ConlluToolkit重写数据加载
            raise RuntimeError(f'当前max_seq_len过小，至少要大于{len(tokens_a) + special_tokens_count},请重新设置')
            # if skip_too_long_input:
            #     # 这里直接跳过过长的句子
            #     skip_input_num += 1
            #     continue
            # else:
            #     tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]
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
        # input_mask:
        #   如果mask_padding_with_zero=True（默认），则 1 代表 实际输入，0 代表 padding
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        start_pos = example.start_pos
        end_pos = example.end_pos
        # print(end_pos)

        if start_pos:
            assert len(start_pos) == len(end_pos)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        position_padding_length = max_seq_length - len(start_pos)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # 由于batched_index_select的限制，position idx的pad不能为-1
            # 如果为0的话，又容易和CLS重复，
            # 所以这里选择用max_seq_length-1表示PAD
            # 注意这里max_seq_length至少应该大于实际句长（以字数统计）3到4个位置
            start_pos = ([max_seq_length - 1] * position_padding_length) + start_pos
            end_pos = ([max_seq_length - 1] * position_padding_length) + end_pos
            assert start_pos[0] == max_seq_length - 1
            assert end_pos[0] == max_seq_length - 1
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            # 由于batched_index_select的限制，position idx的pad不能为-1
            # 如果为0的话，又容易和CLS重复，
            # 所以这里选择用max_seq_length-1表示PAD
            # 注意这里max_seq_length至少应该大于实际句长（以字数统计）3到4个位置
            start_pos = start_pos + ([max_seq_length - 1] * position_padding_length)
            end_pos = end_pos + ([max_seq_length - 1] * position_padding_length)
            assert start_pos[-1] == max_seq_length - 1
            assert end_pos[-1] == max_seq_length - 1
        # 开始位置是ROOT，对应的pos设置为PAD (不计算loss)
        example.pos = ['<PAD>'] + example.pos
        # 如果长度超过max_seq_length，则需要截断
        example.pos = example.pos[:max_seq_length]
        # 将pos序列补足到max_seq_length的长度（用PAD，不计算loss）
        POS_padding_length = max_seq_length - len(example.pos)
        example.pos = example.pos + (['<PAD>'] * POS_padding_length)
        if pos_tokenizer:
            pos_ids = pos_tokenizer.convert_tokens_to_ids(example.pos)
            assert len(pos_ids) == max_seq_length
        else:
            pos_ids = None

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_pos) == len(end_pos) == max_seq_length

        dep_ids = _make_label_target(example.deps, max_seq_length)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          dep_ids=dep_ids,
                          start_pos=start_pos,
                          end_pos=end_pos,
                          pos_ids=pos_ids, ))
    # if skip_input_num > 0:
    #     print(f'\n>> convert_examples_to_features skip input:{skip_input_num} !!\n')
    # else:
    #     print('\n>> No sentences are skipped :)\n')
    return features


def convert_examples_to_features_pool(examples, max_seq_length,
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
                                      mask_padding_with_zero=True,
                                      pos_tokenizer=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    process_fun = partial(one_example_to_feature, max_seq_length=max_seq_length, tokenizer=tokenizer,
                          cls_token_at_end=cls_token_at_end,
                          cls_token=cls_token, cls_token_segment_id=cls_token_segment_id, sep_token=sep_token,
                          sep_token_extra=sep_token_extra, pad_token=pad_token,
                          pad_token_segment_id=pad_token_segment_id,
                          sequence_a_segment_id=sequence_a_segment_id, mask_padding_with_zero=mask_padding_with_zero,
                          pos_tokenizer=pos_tokenizer)
    assert not cls_token_at_end, "CLS必须在句首，目前不支持xlnet"
    assert not pad_on_left, "PAD必须在句子右侧，目前不支持xlnet"
    with Pool(4) as pool:
        print(f'pool._processes:{pool._processes}')
        # Ref:
        # [multithreading - Python 3: does Pool keep the original order of data passed to map? - Stack Overflow]
        # https://stackoverflow.com/questions/41273960/python-3-does-pool-keep-the-original-order-of-data-passed-to-map
        features = pool.map(process_fun, examples)
    return features


def one_example_to_feature(example, max_seq_length,
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
                           mask_padding_with_zero=True,
                           pos_tokenizer=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    assert not cls_token_at_end, "CLS必须在句首，目前不支持xlnet"
    assert not pad_on_left, "PAD必须在句子右侧，目前不支持xlnet"
    assert isinstance(example, InputExample)
    tokens_a = tokenizer.tokenize(example.sentence)
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    # 为ROOT表示等预留足够的空间(目前至少预留5个位置)
    special_tokens_count += 3
    if len(tokens_a) > max_seq_length - special_tokens_count:
        # 由于无论是跳过过长的句子还是截断过长的句子都会导致在dev时无法写入文件，
        # 这里我们暂时不允许过小的max_seq_len
        # todo: 基于ConlluToolkit重写数据加载
        raise RuntimeError(f'当前max_seq_len过小，至少要大于{len(tokens_a) + special_tokens_count},请重新设置')
        # if skip_too_long_input:
        #     # 这里直接跳过过长的句子
        #     skip_input_num += 1
        #     continue
        # else:
        #     tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]
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
    # input_mask:
    #   如果mask_padding_with_zero=True（默认），则 1 代表 实际输入，0 代表 padding
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    start_pos = example.start_pos
    end_pos = example.end_pos
    # print(end_pos)
    if start_pos:
        assert len(start_pos) == len(end_pos)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    position_padding_length = max_seq_length - len(start_pos)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        # 由于batched_index_select的限制，position idx的pad不能为-1
        # 如果为0的话，又容易和CLS重复，
        # 所以这里选择用max_seq_length-1表示PAD
        # 注意这里max_seq_length至少应该大于实际句长（以字数统计）3到4个位置
        start_pos = ([max_seq_length - 1] * position_padding_length) + start_pos
        end_pos = ([max_seq_length - 1] * position_padding_length) + end_pos
        assert start_pos[0] == max_seq_length - 1
        assert end_pos[0] == max_seq_length - 1
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        # 由于batched_index_select的限制，position idx的pad不能为-1
        # 如果为0的话，又容易和CLS重复，
        # 所以这里选择用max_seq_length-1表示PAD
        # 注意这里max_seq_length至少应该大于实际句长（以字数统计）3到4个位置
        start_pos = start_pos + ([max_seq_length - 1] * position_padding_length)
        end_pos = end_pos + ([max_seq_length - 1] * position_padding_length)
        assert start_pos[-1] == max_seq_length - 1
        assert end_pos[-1] == max_seq_length - 1
    # 开始位置是ROOT，对应的pos设置为PAD (不计算loss)
    example.pos = ['<PAD>'] + example.pos
    # 如果长度超过max_seq_length，则需要截断
    example.pos = example.pos[:max_seq_length]
    # 将pos序列补足到max_seq_length的长度（用PAD，不计算loss）
    POS_padding_length = max_seq_length - len(example.pos)
    example.pos = example.pos + (['<PAD>'] * POS_padding_length)
    if pos_tokenizer:
        pos_ids = pos_tokenizer.convert_tokens_to_ids(example.pos)
        assert len(pos_ids) == max_seq_length
    else:
        pos_ids = None
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(start_pos) == len(end_pos) == max_seq_length
    dep_ids = _make_label_target(example.deps, max_seq_length)
    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            dep_ids=dep_ids,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            pos_ids=pos_ids, )
    return feature


class POSTokenizer(object):
    def __init__(self, pos_list: List):
        self.pos_list = ['<PAD>', '<UNK>'] + pos_list

    def convert_tokens_to_ids(self, input_seq: List[str]):
        result = []
        for t in input_seq:
            if t in self.pos_list:
                result.append(self.pos_list.index(t))
            else:
                result.append(self.pos_list.index('<UNK>'))
        return result

    def get_idx(self, token):
        return self.pos_list.index(token)

    def get_label_num(self):
        return len(self.pos_list)


def get_pos_tokenizer(new_pos_list, file_path, merge_train=False, conllu_data=None):
    if conllu_data is not None:
        assert isinstance(conllu_data, CoNLLUData)
    assert isinstance(file_path, pathlib.Path)
    if conllu_data is not None and \
            ((new_pos_list and not merge_train) or
             (new_pos_list and merge_train and not (file_path / 'pos_list.json').exists())):
        # conllu_data不为空
        # 而且
        # [(存在new_pos_list 且 非merge_train) 或者 (存在new_pos_list 且 merge_train 且 不存在pos_list.json)]
        pos_counter = Counter()
        for sent in conllu_data.sentences:
            for word in sent.words:
                pos_counter[word.pos] += 1
        pos_list = list(pos_counter.keys())
        with open(str(file_path / 'pos_list.json'), 'w', encoding='utf-8')as f:
            json.dump(pos_list, f, ensure_ascii=False)
        print('>>> get new pos tokenizer')
    elif (file_path / 'pos_list.json').exists():
        with open(str(file_path / 'pos_list.json'), 'r', encoding='utf-8')as f:
            pos_list = json.load(f)
        print('>>> load pos tokenizer')
    else:
        raise RuntimeError()
    pos_tokenizer = POSTokenizer(pos_list)
    return pos_tokenizer


def load_and_cache_examples(args, conllu_file_path, graph_vocab, tokenizer, training=False):
    logger = get_logger(args.log_name)
    word_vocab = tokenizer.vocab if args.encoder_type == 'bertology' else None
    processor = CoNLLUProcessor(args, graph_vocab, word_vocab)
    label_list = graph_vocab.get_labels()

    if args.use_cache:
        cached_dir, _file_name = pathlib.Path(args.data_dir) / 'cached', pathlib.Path(conllu_file_path).name
        cached_dataset = cached_dir / \
                         f'{_file_name}_{args.encoder_type}_pos-{args.use_pos}_len-{args.max_seq_len}-dataset.torch.cache'
        cached_conllu = cached_dir / \
                        f'{_file_name}-conllu.pickle.cache'
        if not cached_dir.is_dir():
            cached_dir.mkdir()

    if args.use_cache and args.command == 'train' and cached_dataset.is_file():
        # 加载缓存
        logger.info("Loading cached file")
        if cached_conllu.is_file():
            with open(str(cached_conllu), 'rb')as f:
                conllu_file = pickle.load(f)
        else:
            conllu_file, _ = load_conllu_file(conllu_file_path)
        with Timer('Load cached data set'):
            data_set = torch.load(cached_dataset)
        # if args.use_pos:
        #     pos_tokenizer = get_pos_tokenizer(new_pos_list=training, file_path=cached_dir)
        #     args.pos_label_pad_idx = pos_tokenizer.get_idx('<PAD>')
        #     args.pos_label_num = pos_tokenizer.get_label_num()
        return data_set, conllu_file
    else:
        conllu_file, conllu_data = load_conllu_file(conllu_file_path)
        if args.use_pos:
            # 仅在training=True时，生成新的pos_list
            pos_tokenizer = get_pos_tokenizer(new_pos_list=training, file_path=cached_dir, conllu_data=conllu_data)
            args.pos_label_pad_idx = pos_tokenizer.get_idx('<PAD>')
            args.pos_label_num = pos_tokenizer.get_label_num()
        with Timer(f'Create {"train" if training else "dev|infer"} example'):
            examples = processor.create_bert_example(conllu_data,
                                                     'train' if training else 'dev',
                                                     args.max_seq_len,
                                                     training=training,
                                                     )
        with Timer(f'Convert {"train" if training else "dev|infer"} example to features'):
            features = convert_examples_to_features(examples, label_list, args.max_seq_len, tokenizer,
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
                                                    # skip_too_long_input=args.skip_too_long_input,
                                                    pos_tokenizer=pos_tokenizer if args.use_pos else None
                                                    )

        # Convert to Tensors and build dataset
        with Timer(f'{"train" if training else "dev|infer"} Features to Dataset'):
            data_set = feature_to_dataset(features)

        if args.local_rank in [-1, 0] and args.use_cache and args.command == 'train':
            # with Timer(f'Save {"train" if training else "dev|infer"} cache'):
            #     torch.save((conllu_file, features), str(cached_features_file))
            with open(str(cached_conllu), 'wb')as f:
                pickle.dump(conllu_file, f)
            with Timer('Save data set'):
                torch.save(data_set, cached_dataset)
            logger.info("Saved dateset into cached file %s", str(cached_dataset))

    return data_set, conllu_file


def feature_to_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_pos = torch.tensor([t.start_pos for t in features], dtype=torch.long)
    # print([t.end_pos for t in features])
    all_end_pos = torch.tensor([t.end_pos for t in features], dtype=torch.long)
    all_dep_ids = torch.tensor([t.dep_ids for t in features], dtype=torch.long)
    tensors = [all_input_ids, all_input_mask, all_segment_ids, all_start_pos, all_end_pos, all_dep_ids]
    if hasattr(features[0], 'pos_ids'):
        all_pos_ids = torch.tensor([t.pos_ids for t in features], dtype=torch.long)
        tensors.append(all_pos_ids)
    dataset = TensorDataset(*tensors)
    # Input Tensors:
    #   all_input_ids,
    #   all_input_mask,
    #   all_segment_ids,
    #   all_start_pos,
    #   all_end_pos,
    #   all_dep_ids,
    #   all_pos_ids, (如果有)
    return dataset


def get_data_loader(dataset, batch_size, evaluation=False,
                    custom_dataset=False, num_worker=6, local_rank=-1):
    if evaluation:
        sampler = SequentialSampler(dataset)
    else:
        if not custom_dataset:
            # 使用 DistributedSampler 对数据集进行划分
            sampler = RandomSampler(dataset) if local_rank == -1 else DistributedSampler(dataset)
        else:
            sampler = None
    print(f'get_data_loader: training:{not evaluation}; sampler:{sampler}')
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_worker)
    return data_loader


def load_bert_tokenizer(model_path, model_type, do_lower_case=True):
    # 必须把 unused 添加到 additional_special_tokens 上，否则 unused （用来表示ROOT）可能无法正确切分！
    return BERTology_TOKENIZER[model_type].from_pretrained(model_path, do_lower_case=do_lower_case,
                                                           additional_special_tokens=['[unused1]', '[unused2]',
                                                                                      '[unused3]'])


def load_bertology_input(args):
    # todo: 现在没有很好地区分加载不同数据的过程，建议改写为显示输入加载位置，而不是在本程序中根据configs硬编码
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # if args.local_rank not in [-1, 0] and args.run_mode == 'train':
    #     torch.distributed.barrier()
    logger = get_logger(args.log_name)
    logger.info(f'data loader worker num: {args.loader_worker_num}')
    assert (pathlib.Path(args.saved_model_path) / 'vocab.txt').exists()
    tokenizer = load_bert_tokenizer(args.saved_model_path, args.bertology_type)
    vocab = GraphVocab(args.graph_vocab_file)
    if args.command == 'train' and args.local_rank in [-1, 0]:
        tokenizer.save_pretrained(args.output_model_dir)

    if args.command in ['dev', 'infer', 'test_after_train']:
        # training 影响 Input Mask
        logger.info(f'Load data from {args.input_conllu_path}')
        dataset, conllu_file = load_and_cache_examples(args, args.input_conllu_path, vocab, tokenizer, training=False)
        data_loader = get_data_loader(dataset, batch_size=args.eval_batch_size, evaluation=True,
                                      num_worker=args.loader_worker_num)
        return data_loader, conllu_file
    elif args.command == 'train':
        if not args.merge_training:
            train_dataset, train_conllu_file = load_and_cache_examples(args,
                                                                       os.path.join(args.data_dir, args.train_file),
                                                                       vocab, tokenizer, training=True)
        else:
            logger.info(f'merge train: use the ConcatTensorRandomDataset!!!')
            train_text_dataset, _ = load_and_cache_examples(args,
                                                            os.path.join(args.data_dir, args.train_text_file),
                                                            vocab, tokenizer, training=True)
            train_news_dataset, _ = load_and_cache_examples(args,
                                                            os.path.join(args.data_dir, args.train_news_file),
                                                            vocab, tokenizer, training=True)
            train_dataset = ConcatTensorRandomDataset(datasets=[train_text_dataset, train_news_dataset],
                                                      probs=None,
                                                      exp=args.merge_train_exp,
                                                      mode=args.merge_train_mode)
            # 此时无法产生正确的train_conllu_file，不过所幸训练时可以不用train_conllu_file（不过这样就无法计算train metrics了）
            train_conllu_file = None
        train_data_loader = get_data_loader(train_dataset,
                                            batch_size=args.train_batch_size,
                                            evaluation=False,
                                            custom_dataset=args.merge_training,
                                            num_worker=args.loader_worker_num,
                                            local_rank=args.local_rank)

        dev_dataset, dev_conllu_file = load_and_cache_examples(args,
                                                               os.path.join(args.data_dir, args.dev_file),
                                                               vocab, tokenizer, training=False)
        dev_data_loader = get_data_loader(dev_dataset,
                                          batch_size=args.eval_batch_size,
                                          evaluation=True,
                                          num_worker=args.loader_worker_num)
        return train_data_loader, train_conllu_file, dev_data_loader, dev_conllu_file
    else:
        raise RuntimeError('不支持的command {train、dev、infer、test_after_train}')
