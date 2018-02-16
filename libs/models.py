import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, i_size, d, h_size, n_layers=1, bidirec=False,
                 use_cuda=True):
        super(Encoder, self).__init__()
        self.use_cuda = use_cuda
        self.i_size = i_size
        self.h_size = h_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(i_size, d)

        if bidirec:
            self.n_direction = 2
            self.gru = nn.GRU(d,
                              h_size,
                              n_layers,
                              batch_first=True,
                              bidirectional=True)
        else:
            self.n_direction = 1
            self.gru = nn.GRU(d,
                              h_size,
                              n_layers,
                              batch_first=True)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers * self.n_direction,
                                      inputs.size(0),
                                      self.h_size))
        return hidden.cuda() if self.use_cuda else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)

    def forward(self, inputs, input_lens):
        '''
        inputs: B, T (LongTensor)
        input_lens: read lens of input batch (list)
        '''
        hidden = self.init_hidden(inputs)
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded,
                                      input_lens,
                                      batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lens = pad_packed_sequence(outputs, batch_first=True)

        if self.n_layers > 1:
            if self.n_direction == 2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]
        return outputs, torch.cat(list(hidden), 1).unsqueeze(1)


class Decoder(nn.Module):

    def __init__(self, i_size, d, h_size, n_layers=1, dropout_p=0.1,
                 use_cuda=True):
        super(Decoder, self).__init__()
        self.use_cuda = use_cuda
        self.h_size = h_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(i_size, d)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(d + h_size, h_size, n_layers, batch_first=True)
        self.linear = nn.Linear(h_size * 2, i_size)
        self.attn = nn.Linear(self.h_size, self.h_size)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers,
                                      inputs.size(0),
                                      self.h_size))
        return hidden.cuda() if self.use_cuda else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
        self.attn.weight = nn.init.xavier_uniform(self.attn.weight)

    def Attention(self, hidden, encoder_outputs):
        '''
        hidden: 1, B, D
        encoder_outputs: B, T, D
        '''
        hidden = hidden[0].unsqueeze(2)  # (1, B, D) -> (B, D, 1)
        batch_size = encoder_outputs.size(0)  # B
        max_len = encoder_outputs.size(1)
        energeies = self.attn(encoder_outputs.contiguous()
                              .view(batch_size * max_len,
                                    -1))  # (B*T, D) -> (B*T, D)
        energeies = energeies.view(batch_size, max_len, -1)  # (B, T, D)
        # (B, T, D) * (B, D, 1) -> B, T
        attn_energies = energeies.bmm(hidden).squeeze(2)

        alpha = F.softmax(attn_energies)  # B, T
        alpha = alpha.unsqueeze(1)  # B, 1, T
        # (B, 1, T) * (B, T, D) -> (B, 1, D)
        context = alpha.bmm(encoder_outputs)

        return context, alpha

    def forward(self, inputs, context, max_len,
                encoder_outputs, is_training=False):
        '''
        inputs: B, 1 (LongTensor, START SYMBOL)
        context: B, 1, D (FloatTensor, Last encoder hidden state)
        max_len: int, max len to decode  # for batch
        encoder_outputs: B, T, D
        is_training: bool, this is because adapt dropout only trainig step
        '''
        # get the embedding of the current input word
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        if is_training:
            embedded = self.dropout(embedded)

        decode = []
        # apply GRU to the output so far
        for i in range(max_len):
            _, hidden = self.gru(torch.cat((embedded, context), 2),
                                 hidden)  # h_t = f(h_{t-1}, y_{t-1}, c)
            concated = torch.cat((hidden[-1].unsqueeze(0),
                                 context.transpose(0, 1)),
                                 2)  # y_t = g(h_t, y_{t-1}, c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1)  # y_{t-1}
            if is_training:
                embedded = self.dropout(embedded)
            context, alpha = self.Attention(hidden,
                                            encoder_outputs)
            decode.append(softmaxed)
            del softmaxed

        # column-wise concat, reshape
        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0) * max_len, -1)

    def decode(self, context, encoder_outputs, w2i):
        start_decode = Variable(LongTensor([[w2i['<s>']] * 1])) \
            .transpose(0, 1)
        if self.use_cuda:
            start_decode = start_decode.cuda()
        embedded = self.embedding(start_decode)
        hidden = self.init_hidden(start_decode)

        decodes = []
        attentions = []
        decoded = embedded
        for _ in range(50):
            if decoded.data.tolist()[0] == w2i['</s>']:
                break
            _, hidden = self.gru(torch.cat((embedded,
                                            context), 2),
                                 hidden)  # h_t = f(h_{t-1}, y_{t-1}, c)
            concated = torch.cat((hidden[-1].unsqueeze(0),
                                 context.transpose(0, 1)),
                                 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1)
            context, alpha = self.Attention(hidden, encoder_outputs)
            attentions.append(alpha.squeeze(1))

        return torch.cat(decodes).max(1)[1], torch.cat(attentions)
