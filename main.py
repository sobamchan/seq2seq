import os
import sys
from datetime import datetime
from libs.trainer import Trainer
from libs.args import Args
from libs import logger
from libs import utils
import argparse


def train(args):
    trainer = Trainer(args)
    trainer.train()


def test(args):
    trainer = Trainer(args)
    trainer.load_model()
    for pair in trainer.test_data:
        i, t, p = trainer.translate(pair)
        logger.info('--' * 10)
        logger.info('input: {}'.format(i))
        logger.info('truth: {}'.format(t))
        logger.info('predict: {}'.format(p))


def interactive(args):
    trainer = Trainer(args)
    trainer.load_model()
    sent = input('input a sentence: ')
    sent = sent.split() + ['</s>']
    sent = utils.prepare_sequence(sent, trainer.s_w2i, False).view(1, -1)
    src = sent

    sent = ['fake', '</s>']
    sent = utils.prepare_sequence(sent, trainer.s_w2i, False).view(1, -1)
    tgt = sent

    pair = (src, tgt)
    i, t, p = trainer.translate(pair)
    logger.info('--' * 10)
    logger.info('input: {}'.format(i))
    logger.info('predict: {}'.format(p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', help='yaml file path')
    parser.add_argument('-t', default='train', help='type <train or test>')
    parser.add_argument('-o', default=None,
                        help='output dir (default: create with datetime)')
    parser.add_argument('--gpu', dest='g', default=None, type=int,
                        help='gpu to use')
    pargs = parser.parse_args()

    args = Args(pargs.y)

    if pargs.o is None:
        dpath = datetime.now().strftime('%Y%m%d_%H%M%S')
        dpath = os.path.join('experiments', dpath)
        if os.path.exists(dpath):
            print('Directory already exists')
            sys.exit()
        os.mkdir(dpath)
        args.output_dir = dpath
    else:
        args.output_dir = pargs.o

    args.gpu = pargs.g

    with logger.session(args.output_dir):
        if pargs.t == 'train':
            train(args)
        elif pargs.t == 'test':
            test(args)
        elif pargs.t == 'demo':
            interactive(args)
