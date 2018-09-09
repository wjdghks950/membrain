# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.core.torch_agent import Beam
from parlai.core.dict import DictionaryAgent
from .utils.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from .utils.Modules import BottleLinear as Linear
from .utils.Layers import EncoderLayer, DecoderLayer
from .utils.Models import position_encoding_init, get_attn_padding_mask, get_attn_subsequent_mask
from .utils import Constants as Constants
import os
import numpy as np


def pad(tensor, length, dim=0):
    if tensor.size(dim) < length:
        return torch.cat(
            [tensor, tensor.new(*tensor.size()[:dim],
                                length - tensor.size(dim),
                                *tensor.size()[dim+1:]).zero_()],
            dim=dim)
    else:
        return tensor


class Membrain(nn.Module):
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, opt, num_features,
                 padding_idx=0, start_idx=1, end_idx=2, longest_label=1):

        super().__init__()
        self.opt = opt

        self.rank = opt['rank_candidates']
        self.attn_type = opt['attention']

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        rnn_class = Membrain.RNN_OPTS[opt['rnn_class']]
        self.decoder = Decoder(
            num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
            dropout=opt['dropout'], bidir_input=opt['bidirectional'],
            share_output=opt['lookuptable'] in ['dec_out', 'all'],
            attn_type=opt['attention'], attn_length=opt['attention_length'],
            attn_time=opt.get('attention_time'), sparse=False,
            numsoftmax=opt.get('numsoftmax', 1),
            softmax_layer_bias=opt.get('softmax_layer_bias', False),
            num_max_seq=opt['max_seq_len'])

        shared_rnn = self.decoder.rnn if opt['decoder'] == 'shared' else None
        self.encoder = Encoder(
            num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'], dropout=opt['dropout'],
            bidirectional=opt['bidirectional'],
            shared_rnn=shared_rnn, sparse=False,
            num_max_seq=opt['max_seq_len'], num_heads=opt['num_heads'], d_k=opt['d_k'], d_v=opt['d_v'], dim_model=opt['d_model'])

        if self.rank:
            self.ranker = Ranker(
                self.decoder,
                padding_idx=self.NULL_IDX,
                attn_type=opt['attention'])

        self.beam_log_freq = opt.get('beam_log_freq', 0.0)
        if self.beam_log_freq > 0.0:
            self.dict = DictionaryAgent(opt)
            self.beam_dump_filecnt = 0
            self.beam_dump_path = opt['model_file'] + '.beam_dump'
            if not os.path.exists(self.beam_dump_path):
                os.makedirs(self.beam_dump_path)

    def unbeamize_hidden(self, hidden, beam_size, batch_size):
        """
        Creates a view of the hidden where batch axis is collapsed with beam axis,
        we need to do this for batched beam search, i.e. we emulate bigger mini-batch
        :param hidden: hidden state of the decoder
        :param beam_size: beam size, i.e. num of hypothesis
        :param batch_size: number of samples in the mini batch
        :return: view of the hidden
        """
        if isinstance(hidden, tuple):
            num_layers = hidden[0].size(0)
            hidden_size = hidden[0].size(-1)
            return (hidden[0].view(num_layers, batch_size * beam_size, hidden_size),
                hidden[1].view(num_layers, batch_size * beam_size, hidden_size))
        else:  # GRU
            num_layers = hidden.size(0)
            hidden_size = hidden.size(-1)
            return hidden.view(num_layers, batch_size * beam_size, hidden_size)

    def unbeamize_enc_out(self, enc_out, beam_size, batch_size):
        hidden_size = enc_out.size(-1)
        return enc_out.view(batch_size * beam_size, -1, hidden_size)

    #def forward(self, xs, ys=None, cands=None, valid_cands=None, prev_enc=None,
                rank_during_training=False, beam_size=1, topk=1):
        """Get output predictions from the model.

        Arguments:
        xs -- input to the encoder
        ys -- expected output from the decoder
        cands -- set of candidates to rank, if applicable
        valid_cands -- indices to match candidates with their appropriate xs
        prev_enc -- if you know you'll pass in the same xs multiple times and
            the model is in eval mode, you can pass in the encoder output from
            the last forward pass to skip recalcuating the same encoder output
        rank_during_training -- (default False) if set, ranks any available
            cands during training as well
        """
    def forward(self, xs, ys):
        src_seq, src_pos = xs
        tgt_seq, tgt_pos = ys

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))

class EncoderLayer(nn.Module):

    def __init__(self, dim_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
                n_head, dim_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn

class Encoder(nn.Module): #TODO: Implement Encoder based on ""Attention is all you need""
    """Encoder model with a self-attention mechanism"""
    def __init__(self, num_features, padding_idx=0, rnn_class='lstm',
                 emb_size=512, hidden_size=1024, num_layers=6, dropout=0.1,
                 bidirectional=False, shared_rnn=None,
                 sparse=False, num_max_seq=128, num_heads=8, d_k=64, d_v=64, dim_model=512):
        super(Encoder, self).__init__()

        n_position = num_max_seq + 1
        self.num_max_seq = num_max_seq
        self.dim_model = dim_model

        self.position_enc = nn.Embedding(n_position, emb_size, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, emb_size)
        self.src_word_emb = nn.Embedding(num_features, emb_size, padding_idx=Constants.PAD)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim_model, hidden_size, num_heads, d_k, d_v, dropout=dropout)
            for _ in range(num_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        # Word embeding look up
        enc_input = self.src_word_emb(src_seq)



        # Position Encoding attention
        enc_input += self.position_enc(src_pos)


        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)


        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output,
                    slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,


class DecoderLayer(nn.Module):

    def __init__(self, dim_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, dim_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, dim_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
                dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module): #TODO: Implement Decoder based on ""Attention is all you need""
    """Decoder with a self-attention mechanism"""
    def __init__(self, num_features, padding_idx=0, rnn_class='lstm',
                 emb_size=512, hidden_size=1024, dropout=0.1,
                 bidir_input=False, share_output=True,
                 attn_type='none', attn_length=-1, attn_time='pre',
                 sparse=False, numsoftmax=1, softmax_layer_bias=False, num_max_seq=128,
                 num_layers=6, num_heads=8, d_k=64, d_v=64, dim_model=512):

        super(Decoder, self).__init__()
        n_position = num_max_seq + 1
        self.num_max_seq = num_max_seq
        self.dim_model = dim_model

        self.position_enc = nn.Embedding(
                n_position, emb_size, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, emb_size)

        self.tgt_word_emb = nn.Embedding(
                num_features, emb_size, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(dim_model, hidden_size, num_heads, d_k, d_v, dropout=dropout)
            for _ in range(num_layers)]) #TODO: Shouldn't this be copy.deepcopy?

        # MoS - Mixture of Softmaxes
        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.softmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hidden_size, numsoftmax, bias=False)
            self.latent = nn.Linear(hidden_size, numsoftmax * emb_size)
            self.activation = nn.Tanh()

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        # Look up word embedding
        dec_input = self.tgt_word_emb(tgt_seq)

        # Adding positional encoding to the word vector
        dec_input += self.position_enc(tgt_pos)

        # Decode
        self_attn_padmask = get_attn_padding_mask(tgt_seq, tgt_seq)
        self_attn_submask = get_attn_subsequent_mask(tgt_seq)
        self_attnmask = torch.gt(self_attn_padmask + self_attn_submask, 0)

        dec_enc_attn_padmask = get_attn_padding_mask(tgt_seq, src_seq)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output, enc_output,
                    slf_attn_mask=self_attnmask,
                    dec_enc_attn_mask=dec_enc_attn_padmask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        #TODO: Implement MoS(Mixture of Softmaxes) here
        """
        if self.numsoftmax > 1:
        """

        #TODO: Select top scoring index, excluding the padding symbol (at idx zero)
        # we can do topk sampling from renormalized softmax here, default topk=1 is greedy
        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,


class Ranker(object):
    def __init__(self, decoder, padding_idx=0, attn_type='none'):
        super().__init__()
        self.decoder = decoder
        self.NULL_IDX = padding_idx
        self.attn_type = attn_type

    def forward(self, cands, cand_inds, decode_params):
        start, hidden, enc_out, attn_mask = decode_params

        hid, cell = (hidden, None) if isinstance(hidden, torch.Tensor) else hidden
        if len(cand_inds) != hid.size(1):
            cand_indices = start.detach().new(cand_inds)
            hid = hid.index_select(1, cand_indices)
            if cell is None:
                hidden = hid
            else:
                cell = cell.index_select(1, cand_indices)
                hidden = (hid, cell)
            enc_out = enc_out.index_select(0, cand_indices)
            attn_mask = attn_mask.index_select(0, cand_indices)

        cand_scores = []

        for i in range(len(cands)):
            curr_cs = cands[i]

            n_cs = curr_cs.size(0)
            starts = start.expand(n_cs).unsqueeze(1)
            scores = 0
            seqlens = 0
            # select just the one hidden state
            if isinstance(hidden, torch.Tensor):
                nl = hidden.size(0)
                hsz = hidden.size(-1)
                cur_hid = hidden.select(1, i).unsqueeze(1).expand(nl, n_cs, hsz)
            else:
                nl = hidden[0].size(0)
                hsz = hidden[0].size(-1)
                cur_hid = (hidden[0].select(1, i).unsqueeze(1).expand(nl, n_cs, hsz).contiguous(),
                           hidden[1].select(1, i).unsqueeze(1).expand(nl, n_cs, hsz).contiguous())

            cur_enc, cur_mask = None, None
            if attn_mask is not None:
                cur_mask = attn_mask[i].unsqueeze(0).expand(n_cs, attn_mask.size(-1))
                cur_enc = enc_out[i].unsqueeze(0).expand(n_cs, enc_out.size(1), hsz)
            # this is pretty much copied from the training forward above
            if curr_cs.size(1) > 1:
                c_in = curr_cs.narrow(1, 0, curr_cs.size(1) - 1)
                xs = torch.cat([starts, c_in], 1)
            else:
                xs, c_in = starts, curr_cs
            if self.attn_type == 'none':
                preds, score, cur_hid = self.decoder(xs, cur_hid, cur_enc, cur_mask)
                true_score = F.log_softmax(score, dim=2).gather(
                    2, curr_cs.unsqueeze(2))
                nonzero = curr_cs.ne(0).float()
                scores = (true_score.squeeze(2) * nonzero).sum(1)
                seqlens = nonzero.sum(1)
            else:
                for i in range(curr_cs.size(1)):
                    xi = xs.select(1, i)
                    ci = curr_cs.select(1, i)
                    preds, score, cur_hid = self.decoder(xi, cur_hid, cur_enc, cur_mask)
                    true_score = F.log_softmax(score, dim=2).gather(
                        2, ci.unsqueeze(1).unsqueeze(2))
                    nonzero = ci.ne(0).float()
                    scores += true_score.squeeze(2).squeeze(1) * nonzero
                    seqlens += nonzero

            scores /= seqlens  # **len_penalty?
            cand_scores.append(scores)

        max_len = max(len(c) for c in cand_scores)
        cand_scores = torch.cat([pad(c, max_len).unsqueeze(0) for c in cand_scores], 0)
        preds = cand_scores.sort(1, True)[1]
        return preds, cand_scores


class Linear(nn.Module):
    """Custom Linear layer which allows for sharing weights (e.g. with an
    nn.Embedding layer).
    """
    def __init__(self, in_features, out_features, bias=True,
                 shared_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared = shared_weight is not None

        # init weight
        if not self.shared:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            if (shared_weight.size(0) != out_features or
                    shared_weight.size(1) != in_features):
                raise RuntimeError('wrong dimensions for shared weights')
            self.weight = shared_weight

        # init bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.shared:
            # weight is shared so don't overwrite it
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight
        if self.shared:
            # detach weight to prevent gradients from changing weight
            # (but need to detach every time so weights are up to date)
            weight = weight.detach()
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class RandomProjection(nn.Module):
    """Randomly project input to different dimensionality."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features),
                                requires_grad=False)  # fix weights
        self.reset_parameters()

    def reset_parameters(self):
        # experimentally: std=1 appears to affect scale too much, so using 0.1
        self.weight.data.normal_(std=0.1)
        # other init option: set randomly to 1 or -1
        # self.weight.data.bernoulli_(self.weight.fill_(0.5)).mul_(2).sub_(1)

    def forward(self, input):
        return F.linear(input, self.weight)


class AttentionLayer(nn.Module):
    def __init__(self, attn_type, hidden_size, emb_size, bidirectional=False,
                 attn_length=-1, attn_time='pre'):
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hidden_size
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = emb_size
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim,
                                          bias=False)

            if self.attention == 'local':
                # local attention over fixed set of output states
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + input_dim, attn_length, bias=False)
                # combines attention weights with encoder outputs
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                # equivalent to dot if attn is identity
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, enc_out, attn_mask=None):
        if self.attention == 'none':
            return xes

        if type(hidden) == tuple:
            # for lstms use the "hidden" state not the cell state
            hidden = hidden[0]
        last_hidden = hidden[-1]  # select hidden state from last RNN layer

        if self.attention == 'local':
            if enc_out.size(1) > self.max_length:
                offset = enc_out.size(1) - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)
            if attn_weights.size(1) > enc_out.size(1):
                attn_weights = attn_weights.narrow(1, 0, enc_out.size(1))
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                hid = hid.expand(last_hidden.size(0),
                                 enc_out.size(1),
                                 last_hidden.size(1))
                h_merged = torch.cat((enc_out, hid), 2)
                active = F.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                if hid.size(2) != enc_out.size(2):
                    # enc_out has two directions, so double hid
                    hid = torch.cat([hid, hid], 2)
                attn_w_premask = (
                    torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1))
            elif self.attention == 'general':
                hid = self.attn(hid)
                attn_w_premask = (
                    torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1))
            # calculate activation scores
            if attn_mask is not None:
                # remove activation from NULL symbols
                attn_w_premask -= (1 - attn_mask) * 1e20
            attn_weights = F.softmax(attn_w_premask, dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        output = F.tanh(self.attn_combine(merged).unsqueeze(1))
        return output
