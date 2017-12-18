import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from libs import models
from libs import utils
from libs import logger
import pickle


class Trainer(object):

    def __init__(self, args):

        logger.info(args.dump())

        logger.info('initializing')
        self.args = args
        self.use_cuda = torch.cuda.is_available()
        src, tgt = utils.get_corpus('./small_parallel_enja/train.ja',
                                    './small_parallel_enja/train.en')

        # build or load vocab
        if args.load_vocab_dir:
            logger.info('loading vocab from {}'.format(args.load_vocab_dir))
            self.load_vocab()
        else:
            logger.info('builing vocab')
            self.build_vocab(src, tgt)
            logger.info('saving vocab to {}'.format(args.output_dir))
            self.save_vocab()

        logger.info('building dataset')
        self.train_data = utils.build_dataset(src, tgt, self.s_w2i, self.t_w2i)
        src, tgt = utils.get_corpus('./small_parallel_enja/test.ja',
                                    './small_parallel_enja/test.en')
        self.test_data = utils.build_dataset(src, tgt, self.s_w2i, self.t_w2i)

        logger.info('preparing encoder and decoder')
        encoder = models.Encoder(len(self.s_w2i),
                                 args.embedding_size,
                                 args.hidden_size,
                                 args.n_layers,
                                 args.bidirec)
        decoder = models.Decoder(len(self.t_w2i),
                                 args.embedding_size,
                                 args.hidden_size * 2)
        logger.info('initializing weight')
        encoder.init_weight()
        decoder.init_weight()

        if self.use_cuda:
            logger.info('use cuda')
            self.encoder = encoder.cuda()
            self.decoder = decoder.cuda()
        else:
            logger.info('no cuda')
            self.encoder = encoder
            self.decoder = decoder

        logger.info('set loss function and optimizers')
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self.enc_optim = optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.dec_optim = optim.Adam(self.decoder.parameters(), lr=args.lr)

        self.bleus_es = [-10000]
        self.patient_es = 0

        del src, tgt

    def train_one_epoch(self):
        FloatTensor, LongTensor, ByteTensor = utils.get_pytorch_tensors()
        losses = []
        s_w2i = self.s_w2i
        t_w2i = self.t_w2i
        for i, batch in enumerate(utils.get_batch(self.args.batch_size,
                                                  self.train_data)):
            srcs, tgts, s_len, t_len = utils.pad_to_batch(batch, s_w2i, t_w2i)
            input_masks = torch.cat([Variable(ByteTensor(
                tuple(map(lambda s: s == 0, t.data))))
                for t in srcs]).view(srcs.size(0), -1)
            start_decode = Variable(LongTensor([[t_w2i['<s>']] *
                                               tgts.size(0)])).transpose(0, 1)
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            output, hidden_c = self.encoder(srcs, s_len)

            preds = self.decoder(start_decode, hidden_c,
                                 tgts.size(1), output, input_masks, True)

            loss = self.loss_function(preds, tgts.view(-1))
            losses.append(loss.data.tolist()[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 50.0)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 50.0)
            self.enc_optim.step()
            self.dec_optim.step()
        return np.mean(losses)

    def train(self):
        logger.info('start training')
        epoch = self.args.epoch
        for e in range(epoch):
            logger.info('{}th epoch'.format(e))
            loss_mean = self.train_one_epoch()
            bleu_mean = self.calc_bleu()
            logger.logkv('epoch', e)
            logger.logkv('loss_mean', loss_mean)
            logger.logkv('bleu', bleu_mean)
            logger.dumpkvs()
            input_, truth_, pred_ = self.test_translate()
            logger.info('input: {}'.format(input_))
            logger.info('truth: {}'.format(truth_))
            logger.info('pred: {}'.format(pred_))

            self.bleus_es.append(bleu_mean)
            if max(self.bleus_es) == bleu_mean:
                # cool
                pass
            else:
                self.patient_es += 1
                # lame
                if self.patient_es > 3:
                    logger.info('early stopping')
                    break

        logger.info('done training')
        logger.info('saving model to {}'.format(self.args.output_dir))
        self.save_model()

    def translate(self, pair):
        input_ = pair[0]
        truth = pair[1]
        s_i2w = self.s_i2w
        t_i2w = self.t_i2w
        t_w2i = self.t_w2i

        output, hidden = self.encoder(input_, [input_.size(1)])
        pred, attn = self.decoder.decode(hidden, output, t_w2i)

        input_ = [s_i2w[i]
                  for i in input_.data.tolist()[0]
                  if s_i2w[i] not in ['</s>']]
        pred = [t_i2w[i]
                for i in pred.data.tolist()
                if i not in [2, 3]]
        truth = [t_i2w[i]
                 for i in truth.data.tolist()[0]
                 if t_i2w[i] not in ['</s>']]
        return input_, truth, pred

    def test_translate(self):
        pair = random.choice(self.test_data)
        i, t, p = self.translate(pair)
        i = ' '.join(i)
        t = ' '.join(t)
        p = ' '.join(p)
        return i, t, p

    def calc_bleu(self):
        scores = []
        for pair in self.test_data:
            input_, truth, pred = self.translate(pair)
            score = utils.calc_bleu(truth, pred)
            scores.append(score)
        return np.mean(scores)

    def save_model(self):
        save_dir = self.args.output_dir
        encoder_path = os.path.join(save_dir, 'encoder.model')
        decoder_path = os.path.join(save_dir, 'decoder.model')
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def save_vocab(self):
        save_dir = self.args.output_dir
        spath = os.path.join(save_dir, 'src.vocab')
        tpath = os.path.join(save_dir, 'tgt.vocab')
        s_i2w_path = os.path.join(save_dir, 'src_i2w.pkl')
        t_i2w_path = os.path.join(save_dir, 'tgt_i2w.pkl')
        s_w2i_path = os.path.join(save_dir, 'src_w2i.pkl')
        t_w2i_path = os.path.join(save_dir, 'tgt_w2i.pkl')
        with open(spath, 'wb') as f:
            pickle.dump(self.s_vocab, f)
        with open(tpath, 'wb') as f:
            pickle.dump(self.t_vocab, f)
        with open(s_i2w_path, 'wb') as f:
            pickle.dump(self.s_i2w, f)
        with open(t_i2w_path, 'wb') as f:
            pickle.dump(self.t_i2w, f)
        with open(s_w2i_path, 'wb') as f:
            pickle.dump(self.s_w2i, f)
        with open(t_w2i_path, 'wb') as f:
            pickle.dump(self.t_w2i, f)

    def build_vocab(self, src, tgt):
        self.s_vocab, self.t_vocab, self.s_w2i,\
            self.t_w2i, self.s_i2w, self.t_i2w = utils.get_vocab(src,
                                                                 tgt)

    def load_vocab(self):
        load_dir = self.args.load_vocab_dir
        spath = os.path.join(load_dir, 'src.vocab')
        tpath = os.path.join(load_dir, 'tgt.vocab')
        s_i2w_path = os.path.join(load_dir, 'src_i2w.pkl')
        t_i2w_path = os.path.join(load_dir, 'tgt_i2w.pkl')
        s_w2i_path = os.path.join(load_dir, 'src_w2i.pkl')
        t_w2i_path = os.path.join(load_dir, 'tgt_w2i.pkl')
        with open(spath, 'rb') as f:
            self.s_vocab = pickle.load(f)
        with open(tpath, 'rb') as f:
            self.t_vocab = pickle.load(f)
        with open(s_i2w_path, 'rb') as f:
            self.s_i2w = pickle.load(f)
        with open(t_i2w_path, 'rb') as f:
            self.t_i2w = pickle.load(f)
        with open(s_w2i_path, 'rb') as f:
            self.s_w2i = pickle.load(f)
        with open(t_w2i_path, 'rb') as f:
            self.t_w2i = pickle.load(f)

    def load_model(self):
        mdir = self.args.load_model_dir
        logger.info('loading models from {}'.format(mdir))

        encoder_path = os.path.join(mdir, 'encoder.model')
        decoder_path = os.path.join(mdir, 'decoder.model')
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
