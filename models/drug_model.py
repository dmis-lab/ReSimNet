import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import math
import sys
import logging

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.parameter import Parameter

LOGGER = logging.getLogger(__name__)


class DrugModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, drug_embed_dim,
            lstm_layer, lstm_dropout, bi_lstm, linear_dropout, char_vocab_size,
            char_embed_dim, char_dropout, dist_fn, learning_rate,
            binary, is_mlp, weight_decay, is_graph, g_layer,
            g_hidden_dim, g_out_dim, g_dropout):

        super(DrugModel, self).__init__()

        # Save model configs
        self.drug_embed_dim = drug_embed_dim
        self.lstm_layer = lstm_layer
        self.char_dropout = char_dropout
        self.dist_fn = dist_fn
        self.binary = binary
        self.is_mlp = is_mlp
        self.is_graph = is_graph
        self.g_layer = g_layer
        self.g_dropout = g_dropout

        #For rep_idx 4
        if is_graph:
            self.feature_dim = 75
            self.g_hidden_dim = g_hidden_dim
            self.g_out_dim = g_out_dim
            self.weight1 = Parameter(torch.FloatTensor(
                        self.feature_dim, self.g_hidden_dim))
            self.weight2 = Parameter(torch.FloatTensor(
                        self.g_hidden_dim, self.g_hidden_dim))
            self.weight3 = Parameter(torch.FloatTensor(
                        self.g_hidden_dim, self.g_hidden_dim))
            self.weight4 = Parameter(torch.FloatTensor(
                        self.g_hidden_dim, self.g_out_dim))
            #bias : option
            self.bias1 = Parameter(torch.FloatTensor(self.g_hidden_dim))
            self.bias2 = Parameter(torch.FloatTensor(self.g_hidden_dim))
            self.bias3 = Parameter(torch.FloatTensor(self.g_hidden_dim))
            self.bias4 = Parameter(torch.FloatTensor(self.g_out_dim))
            self.init_graph()

        # For rep_idx 0, 1
        elif not is_mlp:
            self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim,
                                           padding_idx=0)
            self.lstm = nn.LSTM(char_embed_dim, drug_embed_dim, lstm_layer,
                                bidirectional=False,
                                batch_first=True, dropout=lstm_dropout)
        # For rep_ix 2, 3
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                #nn.Dropout(0.5),
                nn.ReLU(),
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU(),
                nn.Linear(hidden_dim, drug_embed_dim),
                #nn.Dropout(0.2),
            )
            #self.init_layers()

        # Distance function
        self.dist_fc = nn.Linear(drug_embed_dim, 1)

        # Get params and register optimizer
        info, params = self.get_model_params()
        self.optimizer = optim.Adam(params, lr=learning_rate,
                                    weight_decay=weight_decay)
        # self.optimizer = optim.SGD(params, lr=learning_rate,
        #                            momentum=0.5)
        if binary:
            # self.criterion = nn.BCELoss()
            self.criterion = lambda x, y: y*torch.log(x) + (1-y)*torch.log(1-x)
        else:
            # self.criterion = nn.MSELoss(reduce=False)
            self.criterion = nn.MSELoss()
        LOGGER.info(info)

    def init_graph(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        stdv3 = 1. / math.sqrt(self.weight4.size(1))

        self.weight1.data.uniform_(-stdv1, stdv1)
        self.bias1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        self.bias2.data.uniform_(-stdv2, stdv2)
        self.weight3.data.uniform_(-stdv2, stdv2)
        self.bias3.data.uniform_(-stdv2, stdv2)
        self.weight4.data.uniform_(-stdv3, stdv3)
        self.bias4.data.uniform_(-stdv3, stdv3)

    def init_lstm_h(self, batch_size):
        return (Variable(torch.zeros(
            	self.lstm_layer*1, batch_size, self.drug_embed_dim)).cuda(),
                Variable(torch.zeros(
            	self.lstm_layer*1, batch_size, self.drug_embed_dim)).cuda())

    def init_layers(self):
        nn.init.xavier_normal(self.encoder[0].weight.data)
        nn.init.xavier_normal(self.encoder[2].weight.data)
        # nn.init.xavier_normal(self.encoder[4].weight.data)

    # Set Siamese network as basic LSTM
    def siamese_sequence(self, inputs, length):
        # Character embedding
        c_embed = self.char_embed(inputs)
        # c_embed = F.dropout(c_embed, self.char_dropout)
        maxlen = inputs.size(1)

        if not self.training:
            # Sort c_embed
            _, sort_idx = torch.sort(length, dim=0, descending=True)
            _, unsort_idx = torch.sort(sort_idx, dim=0)
            maxlen = torch.max(length)

            # Pack padded sequence
            c_embed = c_embed.index_select(0, Variable(sort_idx).cuda())
            sorted_len = length.index_select(0, sort_idx).tolist()
            c_packed = pack_padded_sequence(c_embed, sorted_len, batch_first=True)

        else:
            c_packed = c_embed

        # Run LSTM
        init_lstm_h = self.init_lstm_h(inputs.size(0))
        lstm_out, states = self.lstm(c_packed, init_lstm_h)

        hidden = torch.transpose(states[0], 0, 1).contiguous().view(
                                 -1, 1 * self.drug_embed_dim)
        if not self.training:
            # Unsort hidden states
            outputs = hidden.index_select(0, Variable(unsort_idx).cuda())
        else:
            outputs = hidden

        return outputs

    def graph_conv(self, features, adjs):
        weight1 = self.weight1.unsqueeze(0).expand(
                features.size(0), self.weight1.size(0), self.weight1.size(1))
        support1 = torch.bmm(features, weight1)
        layer1 = torch.bmm(adjs, support1)
        layer1_out = F.dropout(F.relu(layer1 + self.bias1),
                self.g_dropout)

        weight2 = self.weight2.unsqueeze(0).expand(
                layer1_out.size(0), self.weight2.size(0), self.weight2.size(1))
        support2 = torch.bmm(layer1_out, weight2)
        layer2 = torch.bmm(adjs, support2)
        layer2_out = F.dropout(F.relu(layer2 + self.bias2),
                self.g_dropout)

        weight3 = self.weight3.unsqueeze(0).expand(
                layer2_out.size(0), self.weight3.size(0), self.weight3.size(1))
        support3 = torch.bmm(layer2_out, weight3)
        layer3 = torch.bmm(adjs, support3)
        layer3_out = F.dropout(F.relu(layer3 + self.bias3),
                self.g_dropout)
        weight4 = self.weight4.unsqueeze(0).expand(
                layer3_out.size(0), self.weight4.size(0), self.weight4.size(1))
        support4 = torch.bmm(layer3_out, weight4)
        layer4 = torch.bmm(adjs, support4)
        layer4_out = layer4 + self.bias4

        graph_conv = F.log_softmax(layer4_out)

        #Choose pooling operation
        pool = nn.MaxPool1d(graph_conv.size(1))
        #pool = nn.AvgPool1d(graph_conv.size(1))
        graph_conv_embed = torch.squeeze(pool(torch.transpose(graph_conv,1,2)))
        return graph_conv_embed


    def siamese_basic(self, inputs):
        return self.encoder(inputs.float())

    def distance_layer(self, vec1, vec2, distance='cos'):
        if distance == 'cos':
            similarity = F.cosine_similarity(
                    vec1 + 1e-16, vec2 + 1e-16, dim=-1)
        elif distance == 'l1':
            similarity = self.dist_fc(torch.abs(vec1 - vec2))
            similarity = similarity.squeeze(1)
        elif distance == 'l2':
            similarity = self.dist_fc(torch.abs(vec1 - vec2) ** 2)
            similarity = similarity.squeeze(1)

        if self.binary:
            similarity = F.sigmoid(similarity)

        return similarity

    def forward(self, key1, key1_len, key2, key2_len, key1_adj, key2_adj):
        if key1_adj is not None and key2_adj is not None:
            embed1 = self.graph_conv(key1, key1_adj)
            embed2 = self.graph_conv(key2, key2_adj)

        elif not self.is_mlp and not self.is_graph:
            embed1 = self.siamese_sequence(key1, key1_len)
            embed2 = self.siamese_sequence(key2, key2_len)

        else:
            embed1 = self.siamese_basic(key1)
            embed2 = self.siamese_basic(key2)

        similarity = self.distance_layer(embed1, embed2, self.dist_fn)
        return similarity, embed1, embed2

    def get_loss(self, outputs, targets):
        if not self.binary:
            loss = self.criterion(outputs, targets)
            # loss = torch.sum(loss * torch.abs(targets)) / loss.size(0)
        else:
            # loss = -1 * self.criterion(outputs, targets)
            # p_t = targets * outputs + (1 - targets) * (1 - outputs)
            # gamma = 2.
            # loss = torch.sum(((1 - p_t) ** gamma) * loss) / loss.size(0)
            loss = self.criterion(outputs, targets)
        return loss

    def get_model_params(self):
        params = []
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())

        return '{}\nparam size: {:,}\n'.format(self, total_size), params

    def save_checkpoint(self, state, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        LOGGER.info('Save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        LOGGER.info('Load checkpoint %s' % filename)
        checkpoint = torch.load(filename)

        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
