import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn import preprocessing
import math
import networkx as nx
from utils.dice import getJaccard_similarity
from utils.process_NFACC import get_adj_lilmatrix
from torch.autograd import Variable
from sklearn.cluster import KMeans
from sklearn import metrics
from layers import GCN, AvgReadout, Discriminator
from models import MLP
from utils.process_NFACC import sparse_mx_to_torch_sparse_tensor
import functools
import operator
import collections
import pandas as pd
import torch.nn.functional as F

from models import DGI_node2, gcn_network2, DGI, DGI1, DGI_network, DGI_node, gcn_network, LogReg  # 0312
from utils import process_NFACC, clustering, RDropLoss
from utils.clustering import convertMatrix_listlabel
import time

# Parameters
def getValue():
    disc = Discriminator(128)
    batch_size = 1
    nb_epochs = 10#90
    # patience = 200
    lr_1 = 0.1
    lr_2 = 0.004
    l2_coef = 0.0
    drop_prob = 0.2
    hid_units = 128  ##这个是输出的维度，也是隐藏层的维度
    dice = 1
    # dice2=1
    # theta = 0.3
    sparse = True
    nonlinearity = 'prelu'
    temperature = 0.02
    bias = 0.0 + 0.02

    global feature_generate_final

    # Datasets
    adj_ori = {}
    cc_label = {}
    sp_adj = {}

    dataset = 'wbn/wbn'
    adj, features, labels, kmeans_labels, idx_train, idx_val, idx_test, graph, numLay, Metric_label, edge = process_NFACC.load_data_our(
        dataset, dice, numLay=5)
    node_num = labels.shape[0]
    labels_ori = labels
    _, k = np.shape(labels)

    for lay in range(numLay):

        features[lay], _ = process_NFACC.preprocess_features(features[lay])
        adj[lay] = process_NFACC.normalize_adj(adj[lay] + sp.eye(adj[lay].shape[0]))

    nb_nodes = features[0].shape[0]
    ft_size = features[0].shape[1]
    nb_classes = labels.shape[1]

    for lay in range(numLay):
        if sparse:
            sp_adj[lay] = process_NFACC.sparse_mx_to_torch_sparse_tensor(adj[lay])
        else:
            adj[lay] = (adj[lay] + sp.eye(adj[lay].shape[0])).todense()

        features[lay] = torch.FloatTensor(
            features[lay][np.newaxis])
        if not sparse:
            adj[lay] = torch.FloatTensor(adj[lay][np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()


    mlp = MLP.MLP(input_dim=128)
    mlp_optimiser = torch.optim.Adam(mlp.parameters(), lr=lr_1, weight_decay=l2_coef)

    model_network = gcn_network2(ft_size, hid_units, nonlinearity, drop_prob)
    network_optimiser = torch.optim.Adam(model_network.parameters(), lr=lr_1, weight_decay=l2_coef)  # 0312

    model_node = DGI_node2(ft_size, hid_units, nonlinearity)
    node_optimiser = torch.optim.Adam(model_node.parameters(), lr=lr_2, weight_decay=l2_coef)


    for epoch in range(1, nb_epochs + 1):
        print("------------epoch {}--------------".format(epoch))
        network_loss = 0
        node_loss = 0
        model_network.train()
        mlp.train()
        model_node.eval()
        network_optimiser.zero_grad()
        mlp_optimiser.zero_grad()
        node_optimiser.zero_grad()
        network_embedding_list = {}
        for lay in range(numLay):
            ret = model_network(features[lay], sp_adj[lay] if sparse else adj[lay], sparse)  # (n, 128)
            network_embedding_list[lay] = ret  # 0312
        embed_sum = torch.zeros_like(network_embedding_list[0])
        for lay in range(numLay):
            embed_sum = embed_sum + network_embedding_list[lay]
        embeds_ave = embed_sum / numLay
        edge_logits, edge_index = mlp(embeds_ave, node_num, edge)
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        edge_tem = torch.sigmoid(gate_inputs)
        edge_generate_before = edge_tem
        edge_generate = edge_generate_before
        feature_generate = process_NFACC.feature_Generate(edge, edge_generate, node_num)
        edge_adj = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])),
                                 shape=(node_num, node_num),
                                 dtype=np.float32)

        edge_adj = sparse_mx_to_torch_sparse_tensor(edge_adj)

        Feature_final = feature_generate[np.newaxis]
        h_2, c = model_node(Feature_final, edge_adj, sparse, None)
        h2_normal = torch.nn.functional.normalize(h_2, p=2, dim=-1, eps=1e-12).squeeze()
        tau = 1.5
        sim = 0
        for lay in range(numLay):
            h_1, c = model_node(features[lay], sp_adj[lay], sparse, None)  #
            h1_normal = torch.nn.functional.normalize(h_1, p=2, dim=-1, eps=1e-12).squeeze()

            s1 = torch.mm(h2_normal, h1_normal.T) / tau
            s1 = torch.softmax(s1, dim=-1)
            cosloss_network = torch.mean(torch.log(torch.diag(s1) + 1e-10))


            loss2 = -cosloss_network
            network_loss = network_loss + loss2

        network_loss.backward()
        network_optimiser.step()  #
        mlp_optimiser.step()

        model_node.train()
        mlp.eval()
        model_network.eval()
        network_embedding_list = {}
        for lay in range(numLay):
            ret = model_network(features[lay], sp_adj[lay] if sparse else adj[lay], sparse)  # (n, 128)
            network_embedding_list[lay] = ret  # 0312

        embed_sum = torch.zeros_like(network_embedding_list[0])
        for lay in range(numLay):
            embed_sum = embed_sum + network_embedding_list[lay]
        embeds_ave = embed_sum / numLay


        edge_logits, edge_index = mlp(embeds_ave, node_num, edge)

        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        edge_tem = torch.sigmoid(gate_inputs)
        edge_generate_before = edge_tem
        edge_generate = edge_generate_before

        feature_generate = process_NFACC.feature_Generate(edge, edge_generate, node_num)
        edge_adj = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])),
                                 shape=(node_num, node_num),
                                 dtype=np.float32)
        # adj = normalize(adj + sp.eye(adj.shape[0]))  # type = scipy.sparse.csr.csr_matrix
        edge_adj = sparse_mx_to_torch_sparse_tensor(edge_adj)  # type = torch.Tensor

        Feature_final = feature_generate[np.newaxis]

        h_2, c = model_node(Feature_final, edge_adj, sparse, None)

        h2_normal = torch.nn.functional.normalize(h_2, p=2, dim=-1, eps=1e-12).squeeze()

        tau = 1.5
        sim = 0
        for lay in range(numLay):
            h_1, c = model_node(features[lay], sp_adj[lay], sparse, None)  #
            h1_normal = torch.nn.functional.normalize(h_1, p=2, dim=-1, eps=1e-12).squeeze()
            s1 = torch.mm(h2_normal, h1_normal.T) / tau
            s1 = torch.softmax(s1, dim=-1)
            cosloss_node = torch.mean(torch.log(torch.diag(s1) + 1e-10))
            loss3 = -cosloss_node
            node_loss = node_loss + loss3
        print("Epoch = {}, network loss = {}, node loss = {}".format(epoch, network_loss, node_loss))
        node_loss.backward()
        node_optimiser.step()
    global output_final
    model_network.eval()
    model_node.eval()
    mlp.eval()
    output_final = 0
    for lay in range(numLay):
        embeds_1, c = model_node(features[lay], sp_adj[lay] if sparse else adj[lay], sparse, None)
        output_final = embeds_1 + output_final
    output_final = output_final / numLay
    kmeans = KMeans(n_clusters=int(k), init='k-means++')
    y_pred = kmeans.fit_predict(output_final.squeeze().detach().numpy())
    clu_label = y_pred

    NMI = metrics.normalized_mutual_info_score(clu_label, Metric_label)
    ARI = metrics.adjusted_rand_score(clu_label, Metric_label)
    print("Final NMI and ARI are {} {} ".format(NMI, ARI))
    return NMI , ARI



a,b=getValue()













