import random
import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def flatten(li):
    return [item for sublist in li for item in sublist]


def get_batch(batch_size, train_data):
    random.shuffle(train_data)
    start_idx = 0
    end_idx = batch_size

    while end_idx < len(train_data):
        batch = train_data[start_idx:end_idx]
        tmp_end_idx = end_idx
        end_idx = end_idx + batch_size
        start_idx = tmp_end_idx
        yield batch

    if end_idx >= len(train_data):
        batch = train_data[start_idx:]
        yield batch


def get_pytorch_tensors(no_cuda=False):
    # check GPUs conditional and return Tensor Clasees
    use_cuda = torch.cuda.is_available()
    if use_cuda and no_cuda is False:
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        ByteTensor = torch.cuda.ByteTensor
    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor
        ByteTensor = torch.ByteTensor
    return FloatTensor, LongTensor, ByteTensor


def pad_to_batch(batch, src_w2i, tgt_w2i, no_cuda=False):
    _, LongTensor, _ = get_pytorch_tensors(no_cuda)
    sorted_batch = sorted(batch,
                          key=lambda b: b[0].size(1),
                          reverse=True)
    x, y = list(zip(*sorted_batch))
    max_x = max([s.size(1) for s in x])
    max_y = max([s.size(1) for s in y])
    src_pad_idx = src_w2i['<PAD>']
    tgt_pad_idx = tgt_w2i['<PAD>']
    x_p, y_p = [], []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            pads = Variable(LongTensor([src_pad_idx]
                                       * (max_x - x[i].size(1)))).view(1, -1)
            x_p.append(torch.cat([x[i], pads], 1))
        else:
            x_p.append(x[i])

        if y[i].size(1) < max_y:
            pads = Variable(LongTensor([tgt_pad_idx]
                                       * (max_y - y[i].size(1)))).view(1, -1)
            y_p.append(torch.cat([y[i], pads], 1))
        else:
            y_p.append(y[i])

    src_var = torch.cat(x_p)
    tgt_var = torch.cat(y_p)
    src_len = [list(map(lambda s: s == src_pad_idx, t.data)).count(False)
               for t in src_var]
    tgt_len = [list(map(lambda s: s == tgt_pad_idx, t.data)).count(False)
               for t in tgt_var]
    return src_var, tgt_var, src_len, tgt_len


def prepare_sequence(seq, w2i, no_cuda):
    _, LongTensor, _ = get_pytorch_tensors(no_cuda)
    idxs = list(map(lambda w: w2i[w] if w in w2i.keys() else w2i['<UNK>'],
                    seq))
    return Variable(LongTensor(idxs))


def get_corpus(src_path, tgt_path, n=None):
    src, tgt = [], []
    with open(src_path, 'r') as f:
        src_r = f.readlines()
    with open(tgt_path, 'r') as f:
        tgt_r = f.readlines()

    if n:
        src_r = src_r[:n]
        tgt_r = tgt_r[:n]

    for s, t in zip(src_r, tgt_r):
        # preprocess
        if s.strip() == '' or t.strip() == '':
            continue
        normalized_s = s.strip().split()
        normalized_t = t.strip().split()
        src.append(normalized_s)
        tgt.append(normalized_t)
    return src, tgt


def get_vocab(src, tgt):
    src_vocab = list(set(flatten(src)))
    tgt_vocab = list(set(flatten(tgt)))

    src_w2i = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    for vo in src_vocab:
        if vo not in src_w2i.keys():
            src_w2i[vo] = len(src_w2i)
    src_i2w = {v: k for k, v in src_w2i.items()}
    tgt_w2i = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    for vo in tgt_vocab:
        if vo not in tgt_w2i.keys():
            tgt_w2i[vo] = len(tgt_w2i)
    tgt_i2w = {v: k for k, v in tgt_w2i.items()}
    return src_vocab, tgt_vocab, src_w2i, tgt_w2i, src_i2w, tgt_i2w


def build_dataset(src, tgt, src_w2i, tgt_w2i, no_cuda=False):
    X_p, y_p = [], []

    for s, t in zip(src, tgt):
        X_p.append(prepare_sequence(s + ['</s>'],
                                    src_w2i, no_cuda).view(1, -1))
        y_p.append(prepare_sequence(t + ['</s>'],
                                    tgt_w2i, no_cuda).view(1, -1))

    return list(zip(X_p, y_p))


def calc_bleu(ref, pred):
    sf = SmoothingFunction().method4
    ref = [ref]
    return sentence_bleu(ref, pred, smoothing_function=sf)
