import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator
import community

import torch.nn.functional as F


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.C = nn.Parameter(torch.FloatTensor(2708, 7))
        self.E = nn.Parameter(torch.FloatTensor(7, 128))
        self.I = torch.eye(7)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.C)
        nn.init.xavier_normal_(self.E)

    def forward(self, cc_label, seq1, seq2, adj, sparse, msk, samp_bias1,
                samp_bias2):
        node_num = seq1.size()[1]
        ret_1 = torch.empty(1, node_num)
        ret_2 = torch.empty(1, node_num)
        h_1 = self.gcn(seq1, adj, sparse)
        print('grad of h_1', h_1.requires_grad)
        h_2 = self.gcn(seq2, adj, sparse)
        for i in range(len(cc_label)):
            h_11 = h_1[0, cc_label[i], :]
            h_22 = h_2[0, cc_label[i], :]
            h_11 = torch.unsqueeze(h_11, 0)
            h_22 = torch.unsqueeze(h_22, 0)
            c = self.read(h_11, msk)
            c = self.sigm(c)

            sc_1, sc_2 = self.disc(c, h_11, h_22, samp_bias1, samp_bias2)

            for p in range(len(cc_label[i])):
                ret_1[0, cc_label[i][p]] = sc_1[0, p]
                ret_2[0, cc_label[i][p]] = sc_2[0, p]

        ret = torch.cat((ret_1, ret_2), 1)
        print('grad of ret', ret.requires_grad)

        return ret

    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)


        return h_1, c


class DGI1(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI1, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.C = nn.Parameter(torch.FloatTensor(2708, 7))
        self.E = nn.Parameter(torch.FloatTensor(7, 128))
        self.I = torch.eye(7)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.C)
        nn.init.xavier_normal_(self.E)

    def forward(self, cc_label, seq1, seq2, adj, sparse, msk, samp_bias1,
                samp_bias2):
        node_num = seq1.size()[1]
        ret_1 = torch.empty(1, node_num)
        ret_2 = torch.empty(1, node_num)
        h_1 = self.gcn(seq1, adj, sparse)


        h_2 = self.gcn(seq2, adj, sparse)
        for i in range(len(cc_label)):
            h_11 = h_1[0, cc_label[i], :]
            h_22 = h_2[0, cc_label[i], :]
            h_11 = torch.unsqueeze(h_11, 0)
            h_22 = torch.unsqueeze(h_22, 0)
            c = self.read(h_11, msk)
            c = self.sigm(c)

            sc_1, sc_2 = self.disc(c, h_11, h_22, samp_bias1, samp_bias2)

            for p in range(len(cc_label[i])):
                ret_1[0, cc_label[i][p]] = sc_1[0, p]
                ret_2[0, cc_label[i][p]] = sc_2[0, p]

        ret = torch.cat((ret_1, ret_2), 1)

        return ret, c, ret_1, ret_2


    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)



        return h_1, c


class DGI_network(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_network, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)  #
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.C = nn.Parameter(torch.FloatTensor(2708, 7))
        self.E = nn.Parameter(torch.FloatTensor(7, 128))
        self.I = torch.eye(7)
        self.init_weight()
    def init_weight(self):
        nn.init.xavier_normal_(self.C)
        nn.init.xavier_normal_(self.E)

    def forward(self, cc_label, seq1, seq2, adj, sparse, msk, samp_bias1,
                samp_bias2):
        node_num = seq1.size()[1]
        ret_1 = torch.empty(1, node_num)
        ret_2 = torch.empty(1, node_num)
        h_1 = self.gcn(seq1, adj, sparse)
        h_2 = self.gcn(seq2, adj, sparse)
        for i in range(len(cc_label)):
            h_11 = h_1[0, cc_label[i], :]
            h_22 = h_2[0, cc_label[i], :]
            h_11 = torch.unsqueeze(h_11, 0)
            h_22 = torch.unsqueeze(h_22, 0)
            c = self.read(h_11, msk)
            c = self.sigm(c)

            sc_1, sc_2 = self.disc(c, h_11, h_22, samp_bias1, samp_bias2)

            for p in range(len(cc_label[i])):
                ret_1[0, cc_label[i][p]] = sc_1[0, p]
                ret_2[0, cc_label[i][p]] = sc_2[0, p]

        ret = torch.cat((ret_1, ret_2), 1)
        return ret, h_1, h_2


    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)



        return h_1, c
    #print params
    def outputparameter(self):
        print("run output_parameter")
        print(type(self.named_parameters()))
        # 遍历模型参数
        for name, param in self.named_parameters():
            print(name, param.size())



class DGI_node(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_node, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.C = nn.Parameter(torch.FloatTensor(2708, 7))
        self.E = nn.Parameter(torch.FloatTensor(7, 128))
        self.I = torch.eye(7)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.C)
        nn.init.xavier_normal_(self.E)

    def forward(self, cc_label, seq1, seq2, adj, sparse, msk, samp_bias1,
                samp_bias2):

        h_1 = self.gcn(seq1, adj, sparse)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse)

        sc_1, sc_2 = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        ret = torch.cat((sc_1, sc_2), 1)
        return ret, h_1, h_2

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        # print("h_1的size为")


        return h_1, c

class GCN_network(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_network, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_node(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_node, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class gcn_network(nn.Module):
    def __init__(self, n_in, n_h, activation, dropout):
        super(gcn_network, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, 128, activation)
        self.drop = dropout

    def forward(self, seq1, adj, sparse):
        node_num = seq1.size()[1]
        ret_1 = self.gcn(seq1, adj, sparse)
        ret_1 = F.dropout(ret_1, self.drop, training=self.training)
        ret11 = self.gcn2(ret_1, adj, sparse)

        return ret11

    def outputparameter(self):
        print("run output_parameter")
        print(type(self.named_parameters()))
        # 遍历模型参数
        for name, param in self.named_parameters():
            print(name, param.size())