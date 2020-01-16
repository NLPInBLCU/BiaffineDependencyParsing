# -*- coding: utf-8 -*-
# Created by li huayong on 2019/9/24
import os
import pathlib
import torch
from utils.input_utils.bertology.input_class import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers import BertTokenizer, RobertaTokenizer, XLMTokenizer, XLNetTokenizer
from utils.input_utils.custom_dataset import ConcatTensorRandomDataset
from utils.logger import get_logger

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
                                 skip_too_long_input=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    assert not cls_token_at_end, "CLS必须在句首，目前不支持xlnet"
    assert not pad_on_left, "PAD必须在句子右侧，目前不支持xlnet"
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
        # 为ROOT表示等预留足够的空间(目前至少预留5个位置)
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
            # 由于batched_index_select的限制，position idx的pad不能为-1
            # 如果为0的话，又容易和CLS重复，
            # 所以这里选择用max_seq_length-1表示PAD
            # 注意这里max_seq_length至少应该大于实际句长（以字数统计）3到4个位置
            start_pos = ([max_seq_length - 1] * pos_padding_length) + start_pos
            end_pos = ([max_seq_length - 1] * pos_padding_length) + end_pos
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
            start_pos = start_pos + ([max_seq_length - 1] * pos_padding_length)
            end_pos = end_pos + ([max_seq_length - 1] * pos_padding_length)
            assert start_pos[-1] == max_seq_length - 1
            assert end_pos[-1] == max_seq_length - 1
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
        print(f'\n>> convert_examples_to_features skip input:{skip_input_num} !!\n')
    else:
        print('\n>> No sentences are skipped :)\n')
    return features


def load_and_cache_examples(args, conllu_file_path, graph_vocab, tokenizer, training=False):
    logger = get_logger(args.log_name)
    word_vocab = tokenizer.vocab if args.encoder_type == 'bertology' else None
    processor = CoNLLUProcessor(args, graph_vocab, word_vocab)

    label_list = graph_vocab.get_labels()
    _cwd, _file_name = pathlib.Path(conllu_file_path).cwd(), pathlib.Path(conllu_file_path).name
    cached_features_file = _cwd / (
        f'cached_{_file_name}_{"train" if training else "dev"}_{pathlib.Path(args.config_file).name}_{args.encoder_type}')
    if cached_features_file.is_file() and args.use_cache:
        logger.info("Loading features from cached file %s", str(cached_features_file))
        conllu_file, features = torch.load(str(cached_features_file))
    else:
        conllu_file, conllu_data = load_conllu_file(conllu_file_path)
        examples = processor.create_bert_example(conllu_data,
                                                 'train' if training else 'dev',
                                                 args.max_seq_len,
                                                 training=training,
                                                 )
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
                                                skip_too_long_input=args.skip_too_long_input,
                                                )
        if args.local_rank in [-1, 0] and args.use_cache:
            logger.info("Saving features into cached file %s", str(cached_features_file))
            torch.save((conllu_file, features), str(cached_features_file))
    # Convert to Tensors and build dataset
    data_set = feature_to_dataset(features)
    return data_set, conllu_file


def feature_to_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_pos = torch.tensor([t.start_pos for t in features], dtype=torch.long)
    # print([t.end_pos for t in features])
    all_end_pos = torch.tensor([t.end_pos for t in features], dtype=torch.long)
    all_dep_ids = torch.tensor([t.dep_ids for t in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_pos, all_end_pos, all_dep_ids)
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
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # if args.local_rank not in [-1, 0] and args.run_mode == 'train':
    #     torch.distributed.barrier()
    logger = get_logger(args.log_name)
    logger.info(f'data loader worker num: {args.loader_worker_num}')
    assert (pathlib.Path(args.saved_model_path) / 'vocab.txt').exists()
    tokenizer = load_bert_tokenizer(args.saved_model_path, args.bertology_type)
    vocab = GraphVocab(args.graph_vocab_file)
    if args.run_mode == 'train' and args.local_rank in [-1, 0]:
        tokenizer.save_pretrained(args.output_model_dir)

    if args.run_mode in ['dev', 'inference']:
        # training 影响 Input Mask
        dataset, conllu_file = load_and_cache_examples(args, args.input_conllu_path, vocab, tokenizer, training=False)
        data_loader = get_data_loader(dataset, batch_size=args.eval_batch_size, evaluation=True,
                                      num_worker=args.loader_worker_num)
        return data_loader, conllu_file
    elif args.run_mode == 'train':
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
