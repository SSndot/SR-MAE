#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from params import args


class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        hidden_his = []
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
            hidden_his.append(hidden)
        return hidden, hidden_his


class SessionGraph(nn.Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden, hidden_his = self.gnn(A, hidden)
        return hidden, hidden_his


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class Encoder:
    def __init__(self):
        self.alias_inputs = torch.Tensor().long()
        self.items = torch.Tensor().long()
        self.A = torch.Tensor().float()
        self.mask = torch.Tensor().long()
        self.targets = []

    def forward(self, model, i, data, edge_mask=None):
        alias_inputs, A, items, mask, targets = data.get_slice(i, edge_mask)
        self.alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        self.items = trans_to_cuda(torch.Tensor(items).long())
        self.A = trans_to_cuda(torch.Tensor(A).float())
        self.mask = trans_to_cuda(torch.Tensor(mask).long())
        self.targets = targets
        hidden, hidden_his = model(self.items, self.A)
        return hidden, hidden_his

    def compute(self, model, hidden):
        get = lambda i: hidden[i][self.alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(self.alias_inputs)).long()])
        return self.targets, model.compute_scores(seq_hidden, self.mask)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(args.latdim * args.num_gcn_layers ** 2, args.latdim * args.num_gcn_layers, bias=True),
            nn.ReLU(),
            nn.Linear(args.latdim * args.num_gcn_layers, args.latdim, bias=True),
            nn.ReLU(),
            nn.Linear(args.latdim, 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        # pos: (batch, 2), neg: (batch, num_reco_neg, 2)
        pos_emb, neg_emb = [], []
        for i in range(args.num_gcn_layers):
            for j in range(args.num_gcn_layers):
                pos_emb.append(embeds[i][pos[:,0]] * embeds[j][pos[:,1]])
                neg_emb.append(embeds[i][neg[:,:,0]] * embeds[j][neg[:,:,1]])
        pos_emb = torch.cat(pos_emb, -1) # (n, latdim * num_gcn_layers ** 2)
        neg_emb = torch.cat(neg_emb, -1) # (n, num_reco_neg, latdim * num_gcn_layers ** 2)
        pos_scr = torch.exp(torch.squeeze(self.MLP(pos_emb))) # (n)
        neg_scr = torch.exp(torch.squeeze(self.MLP(neg_emb))) # (n, num_reco_neg)
        neg_scr = torch.sum(neg_scr, -1) + pos_scr
        loss = -torch.sum(pos_scr / (neg_scr + 1e-8) + 1e-8)
        return loss

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


def train_test(model, train_data, test_data):
    encoder = Encoder()
    decoder = Decoder()
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    decoder.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        edge_mask = train_data.generate_mask(i)
        hidden, hidden_his = encoder.forward(model, i, train_data)

        targets, scores = encoder.compute(model, hidden)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)

        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        hidden, hidden_his = encoder.forward(model, i, train_data)

        targets, scores = encoder.compute(model, hidden)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
