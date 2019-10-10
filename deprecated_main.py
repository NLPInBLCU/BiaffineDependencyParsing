import os
import numpy as np
import random
import torch
from torch import optim
from tensorboardX import SummaryWriter

from utils.arguments_old import parse_args
from utils.input_utils.common import DataLoader
from models.sdp_biaffine_trainer import Trainer
from utils.model_utils import sdp_simple_scorer
from utils.input_utils.common import Pretrain
from utils.path import ensure_dir
from utils.timer import Timer
from utils.logger import init_logger


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        logger.critical('use CPU !')
        args.cuda = False
    elif args.cuda:
        logger.critical("use CUDA")
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    logger.info('model args:')
    for k, v in args.items():
        logger.info(f'--{k}: {v}')
    logger.critical(f"Running parser in {args['mode']} mode")

    if args['mode'] == 'train':
        with Timer('train time:'):
            train(args)
    else:
        with Timer('predict time:'):
            evaluate(args)


def train(args):
    ensure_dir(args['save_dir'])
    model_text_file = os.path.join(args['save_dir'], 'TEXT_' + args['save_name_suffix'] + '.pt')
    model_news_file = os.path.join(args['save_dir'], 'NEWS_' + args['save_name_suffix'] + '.pt')

    # load pretrained vectors
    pretrain = Pretrain(args['vec_file'], args['logger_name'])
    # TensorboardX:
    summary_writer = SummaryWriter()

    # load data
    logger.critical(f"Loading data with batch size {args['batch_size']}...")
    train_batch = DataLoader(args['train_merge_file'], args['batch_size'], args, pretrain, evaluation=False)
    vocab = train_batch.vocab
    dev_text_batch = DataLoader(args['dev_text_file'], 1000, args, pretrain, vocab=vocab, evaluation=True)
    dev_news_batch = DataLoader(args['dev_news_file'], 1000, args, pretrain, vocab=vocab, evaluation=True)

    # pred and gold path
    output_text_file = os.path.join(args['output_file_path'], 'DEV_Text_' + args['save_name_suffix'] + '.conllu')
    output_news_file = os.path.join(args['output_file_path'], 'DEV_News_' + args['save_name_suffix'] + '.conllu')

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_text_batch) == 0 or len(dev_news_batch) == 0:
        logger.info("Skip training because no data available...")
        exit()

    logger.info("Training parser...")
    trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'])

    global_step = 0

    using_amsgrad = False
    task2idx = {'text': 0, 'news': 1}
    last_best_step = [0] * 2
    # start training
    train_loss = 0
    last_best_LAS = [0] * 2
    last_best_UAS = [0] * 2
    last_best_LAS_step = [0] * 2
    last_best_UAS_step = [0] * 2

    def _dev(dev_name):
        nonlocal last_best_LAS, last_best_UAS, last_best_LAS_step, last_best_UAS_step
        nonlocal train_loss, last_best_step, dev_text_batch, dev_news_batch
        # nonlocal output_news_file, output_text_file
        if dev_name == 'text':
            dev_batch = dev_text_batch
            output_file = output_text_file
            gold_file = args['dev_text_file']
            task_id = task2idx['text']
            model_file = model_text_file
        elif dev_name == 'news':
            dev_batch = dev_news_batch
            output_file = output_news_file
            gold_file = args['dev_news_file']
            task_id = task2idx['news']
            model_file = model_news_file
        else:
            raise ValueError('bad dev name')
        dev_preds = []
        for batch in dev_batch:
            preds = trainer.predict(batch, cuda_data=False)
            dev_preds += preds
            if args['cuda']:
                torch.cuda.empty_cache()
        dev_batch.conll.set(['deps'], [y for x in dev_preds for y in x])
        dev_batch.conll.write_conll(output_file)
        dev_uas, dev_score = sdp_simple_scorer.score(output_file, gold_file)
        logger.info(
            f"step:{global_step}; train_loss:{train_loss:0.5f}; dev_UAS:{dev_uas:.4f}; dev_LAS:{dev_score:.4f}")
        summary_writer.add_scalars(f'data/eval_{dev_name}',
                                   {'ave_loss': train_loss,
                                    'dev_UAS': dev_uas,
                                    'dev_LAS': dev_score},
                                   global_step)
        # train_loss = 0
        if dev_uas > last_best_UAS[task_id]:
            last_best_UAS[task_id] = dev_uas
            last_best_UAS_step[task_id] = global_step
        # save best model
        if dev_score > last_best_LAS[task_id]:
            last_best_LAS[task_id] = dev_score
            last_best_LAS_step[task_id] = global_step
            last_best_step[task_id] = global_step
            trainer.save(model_file)
            logger.info(f"{dev_name}: last_best_UAS:{last_best_UAS[task_id]:.4f} in step:{last_best_UAS_step[task_id]}")
            logger.critical(
                f"{dev_name}: last_best_LAS:{last_best_LAS[task_id]:.4f} in step:{last_best_LAS_step[task_id]}")
            logger.info(f"{dev_name}: new best model saved in {model_file}")

    while True:
        do_break = False
        for i, batch in enumerate(train_batch):
            # pprint(batch[0].size())
            # continue
            global_step += 1
            print('batch:')
            print(batch)
            loss = trainer.update(batch, global_step, cuda_data=False, eval=False)  # update step
            train_loss += loss
            if args['cuda']:
                torch.cuda.empty_cache()
            if global_step % args['log_step'] == 0:
                summary_writer.add_scalar('data/loss', loss, global_step)

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                # logger.info("Evaluating on dev set...")
                train_loss = train_loss / args['eval_interval']  # avg loss per batch
                _dev('text')
                _dev('news')
                train_loss = 0
            if global_step - max(last_best_step) >= args['max_steps_before_stop']:
                if not using_amsgrad:
                    logger.critical(f"--->>> Optim:Switching to AMSGrad in step:{global_step}")
                    last_best_step = [global_step] * 2
                    using_amsgrad = True
                    trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'],
                                                   betas=(args['beta1'], args['beta2']), eps=args['eps'])
                else:
                    do_break = True
                    break

            if global_step >= args['max_steps']:
                do_break = True
                break

        if do_break:
            break

        train_batch.reshuffle()

    logger.critical(f"Training ended with {global_step} steps")
    logger.critical(f'Text: best dev LAS:{last_best_LAS[0]} in step:{last_best_LAS_step[0]}')
    logger.critical(f'News: best dev LAS:{last_best_LAS[1]} in step:{last_best_LAS_step[1]}')
    summary_writer.close()


def evaluate(args):
    ensure_dir(args['save_dir'])
    model_text_file = os.path.join(args['save_dir'], 'TEXT_' + args['save_name_suffix'] + '.pt')
    model_news_file = os.path.join(args['save_dir'], 'NEWS_' + args['save_name_suffix'] + '.pt')
    pretrain = Pretrain(args['vec_file'], logger_name='evaluate')
    use_cuda = args['cuda'] and not args['cpu']

    def _eval(eval_name):
        if eval_name == 'text':
            model_file = model_text_file
            test_file = args['test_text_file']
            output_file = os.path.join(args['output_file_path'],
                                       'TEST_Text_' + args['save_name_suffix'] + '.conllu')
            gold_file = args['test_text_file']
        elif eval_name == 'news':
            model_file = model_news_file
            test_file = args['test_news_file']
            output_file = os.path.join(args['output_file_path'],
                                       'TEST_News_' + args['save_name_suffix'] + '.conllu')
            gold_file = args['test_news_file']
        else:
            raise ValueError('bad eval name')
        trainer = Trainer(pretrain=pretrain, model_file=model_file, use_cuda=use_cuda)
        loaded_args, vocab = trainer.args, trainer.vocab
        test_batch = DataLoader(test_file, 1000, args, pretrain, vocab=vocab, evaluation=True)
        if len(test_batch) > 0:
            print(f"{eval_name} Start evaluation...")
            preds = []
            for i, b in enumerate(test_batch):
                preds += trainer.predict(b)
        else:
            preds = []
        test_batch.conll.set(['deps'], [y for x in preds for y in x])
        test_batch.conll.write_conll(output_file)
        if gold_file is not None:
            UAS, LAS = sdp_simple_scorer.score(output_file, gold_file)
            print(f"{eval_name}: Test Dataset Parser score:")
            print(f'{eval_name} LAS:{LAS * 100:.3f}\t{eval_name} UAS:{UAS * 100:.3f}')

    _eval('text')
    _eval('news')


if __name__ == '__main__':
    args = parse_args()
    args.cpu = True
    # args.self_att = True
    # args.vec_file = 'Embeds/sem16_tencent.pkl'
    args.batch_size = 30
    args.rec_dropout = 0
    args.char_rec_dropout = 0
    args.num_layers = 1

    logger = init_logger(args.logger_name, args.log_file, is_debug=False)
    logger.critical('######----------->>>>main<<<<-----------######')
    # logger.info(f"args:{args}")
    main(args)
