import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
#from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import Counter
import community as community_louvain
from utils.dice import getSimilariy_modified
from utils.dice import getJaccard_similarity
from scipy.sparse import csr_matrix

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))

    preds = preds.long()
    labels = labels.long()

    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_multilayerNet(dataset_str, dice, numlay):
    Feature = {}
    Adj = {}
    G = {}
    Sim = {}
    Kmeans_F = {}

    print('Loading {} dataset...'.format(dataset_str))
    labels = np.loadtxt(
    "./data/"
     + dataset_str + "_true_idx.txt",
        dtype='int')
    Metric_label = labels
    labels = get_labelmatrix(labels)

    node_num = labels.shape[0]
    labels_num = labels.shape[1]
    idx_train, idx_val, idx_test = get_vttdata(node_num)
    numLay = numlay
    edge_index_total = {}
    for l in range(numLay):
        adj = get_adj_lilmatrix(
            "./data/" + dataset_str + "_Adj_Layer" + str(
                l + 1) + ".txt", node_num)
        edge_index = np.loadtxt(
            "./data/" + dataset_str + "_Adj_Layer" + str(
                l + 1) + ".txt")
        features = np.loadtxt(
            "./data/" + dataset_str + "_Feature_Layer" + str(
                l + 1) + ".txt")
        features = sp.lil_matrix(features)
        Feature[l] = features
        print("features is.....{}".format(features.shape))
        g = nx.from_scipy_sparse_matrix(adj)
        G[l] = g
        I = np.eye(adj.shape[0])
        adj = adj + dice * I
        Adj[l] = adj
        kmeans_features_labels = kmeans(labels_num, features)
        Kmeans_F[l] = kmeans_features_labels
        edge_index_total[l] = edge_index #存了所有边
    edge = edge_index_total[0]
    for lay in range(1, numLay):
        edge = np.vstack([edge, edge_index_total[lay]])
    edge = np.unique(edge, axis=0) #去除不同网络层中的相同连接
    return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge
   ###读取函数结束

    if dataset_str == "citeseer":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/citeseer/citeseer_true_idx.txt",dtype='int')
        Metric_label = labels
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 2
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/citeseer/citeseer_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/citeseer/citeseer_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/citeseer/citeseer_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)
        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "mit":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/mit/mit_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 2
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/mit/mit_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/mit/mit_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/mit/mit_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)
        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "WTN":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/WTN/WTN_new_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 14
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/WTN/WTN_new_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/WTN/WTN_new_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/WTN/WTN_new_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)

        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "cora3":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/cora3/cora_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/cora3/cora_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/cora3/cora_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/cora3/cora_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis = 0)


        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "wbn":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/wbn/wbn_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/wbn/wbn_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/wbn/wbn_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/wbn/wbn_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis = 0)


        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "snd":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/snd/snd_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/snd/snd_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/snd/snd_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/snd/snd_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis = 0)


        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "128-0.5-0.2":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/128/128-0.5-0.2_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/128/128-0.5-0.2_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/128/128-0.5-0.2_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/128/128-0.5-0.2_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis = 0)


        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "128-0.5-0.3":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/128/128-0.5-0.3_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/128/128-0.5-0.3_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/128/128-0.5-0.3_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/128/128-0.5-0.3_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis = 0)


        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "128-0.5-0.4":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/128/128-0.5-0.4_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/128/128-0.5-0.4_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/128/128-0.5-0.4_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/128/128-0.5-0.4_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis = 0)


        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "128-0.5-0.5":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/128/128-0.5-0.5_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix("./data/128/128-0.5-0.5_Adj_Layer"+str(l+1)+".txt", node_num)
            edge_index = np.loadtxt("./data/128/128-0.5-0.5_Adj_Layer"+str(l+1)+".txt")
            features = np.loadtxt("./data/128/128-0.5-0.5_Feature_Layer"+str(l+1)+".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] =  kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis = 0)


        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "128-0.5-0.6":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt(
            "./data/128/128-0.5-0.6_true_idx.txt",
            dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix(
                "./data/128/128-0.5-0.6_Adj_Layer" + str(
                    l + 1) + ".txt", node_num)
            edge_index = np.loadtxt(
                "./data/128/128-0.5-0.6_Adj_Layer" + str(
                    l + 1) + ".txt")
            features = np.loadtxt(
                "./data/128/128-0.5-0.6_Feature_Layer" + str(
                    l + 1) + ".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] = kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)

        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "128-0.5-0.7":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt(
            "./data/128/128-0.5-0.7_true_idx.txt",
            dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix(
                "./data/128/128-0.5-0.7_Adj_Layer" + str(
                    l + 1) + ".txt", node_num)
            edge_index = np.loadtxt(
                "./data/128/128-0.5-0.7_Adj_Layer" + str(
                    l + 1) + ".txt")
            features = np.loadtxt(
                "./data/128/128-0.5-0.7_Feature_Layer" + str(
                    l + 1) + ".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] = kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)

        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "128-0.5-0.8":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt(
            "./data/128/128-0.5-0.8_true_idx.txt",
            dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix(
                "./data/128/128-0.5-0.8_Adj_Layer" + str(
                    l + 1) + ".txt", node_num)
            edge_index = np.loadtxt(
                "./data/128/128-0.5-0.8_Adj_Layer" + str(
                    l + 1) + ".txt")
            features = np.loadtxt(
                "./data/128/128-0.5-0.8_Feature_Layer" + str(
                    l + 1) + ".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            similarities = getJaccard_similarity(labels.shape[0], g)
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] = kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)

        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "128-0.5-0.2-3":
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt(
            "./data/128Layer/128-0.5-0.2-3_true_idx.txt",
            dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = get_adj_lilmatrix(
                "./data/128Layer/128-0.5-0.2-3_Adj_Layer" + str(
                    l + 1) + ".txt", node_num)
            edge_index = np.loadtxt(
                "./data/128Layer/128-0.5-0.2-3_Adj_Layer" + str(
                    l + 1) + ".txt")
            features = np.loadtxt(
                "./data/128Layer/128-0.5-0.2-3_Feature_Layer" + str(
                    l + 1) + ".txt")
            features = sp.lil_matrix(features)
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            g = nx.from_scipy_sparse_matrix(adj)
            G[l] = g
            I = np.eye(adj.shape[0])
            adj = adj + dice * I
            Adj[l] = adj
            kmeans_features_labels = kmeans(labels_num, features)
            Kmeans_F[l] = kmeans_features_labels
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)

        return Adj, Feature, labels, Kmeans_F, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge


def load_multilayerNet_GCN(dataset_str):
    if dataset_str == "cora3":
        Feature = {}
        Adj = {}
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/cora3/cora_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = np.genfromtxt("./data/cora3/cora_Adj_Layer"+str(l+1)+".txt", dtype=np.int32)
            adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),
                                shape=(node_num, node_num),
                                dtype=np.float32)
            # adj = normalize(adj + sp.eye(adj.shape[0]))  # shape= （2708， 2708）， type = scipy.sparse.csr.csr_matrix
            adj = sparse_mx_to_torch_sparse_tensor(adj)  # shape =torch.Size([2708, 2708], type = torch.Tensor

            edge_index = np.loadtxt("./data/cora3/cora_Adj_Layer"+str(l+1)+".txt")
            features = np.genfromtxt("./data/cora3/cora_Feature_Layer"+str(l+1)+".txt")
            features = sp.csr_matrix(features, dtype=np.float32)  #type=scipy.sparse.csr.csr_matrix,
            features = normalize(features)
            features = torch.FloatTensor(
                np.array(features.todense()))
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            Adj[l] = adj
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)

        Kmeans_F = 0
        G = 0
        kmeans_features_labels = 0
        return Adj, Feature, labels, kmeans_features_labels, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    def load_multilayerNet_GCN(dataset_str):
        if dataset_str == "wbn":
            Feature = {}
            Adj = {}
            print('Loading {} dataset...'.format(dataset_str))
            labels = np.loadtxt(
                "./data/cora3/cora_true_idx.txt",
                dtype='int')
            Metric_label = labels
            labels = get_labelmatrix(labels)
            # labels = get_labelmatrix(labels)

            node_num = labels.shape[0]
            labels_num = labels.shape[1]
            idx_train, idx_val, idx_test = get_vttdata(node_num)
            numLay = 3
            edge_index_total = {}
            for l in range(numLay):
                adj = np.genfromtxt(
                    "./data/wbn/wbn_Adj_Layer" + str(
                        l + 1) + ".txt", dtype=np.int32)
                adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),
                                    shape=(node_num, node_num),
                                    dtype=np.float32)
                # adj = normalize(adj + sp.eye(adj.shape[0]))  # shape= （2708， 2708）， type = scipy.sparse.csr.csr_matrix
                adj = sparse_mx_to_torch_sparse_tensor(adj)  # shape =torch.Size([2708, 2708], type = torch.Tensor

                edge_index = np.loadtxt(
                    "./data/wbn/wbn_Adj_Layer" + str(
                        l + 1) + ".txt")
                features = np.genfromtxt(
                    "./data/wbn/wbn_Feature_Layer" + str(
                        l + 1) + ".txt")
                features = sp.csr_matrix(features, dtype=np.float32)  # type=scipy.sparse.csr.csr_matrix,
                features = normalize(features)
                features = torch.FloatTensor(
                    np.array(features.todense()))
                Feature[l] = features
                print("features is.....{}".format(features.shape))
                Adj[l] = adj
                edge_index_total[l] = edge_index
            edge = edge_index_total[0]
            for lay in range(1, numLay):
                edge = np.vstack([edge, edge_index_total[lay]])
            edge = np.unique(edge, axis=0)

            Kmeans_F = 0
            G = 0
            kmeans_features_labels = 0
            return Adj, Feature, labels, kmeans_features_labels, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

    if dataset_str == "snd":
        Feature = {}
        Adj = {}
        print('Loading {} dataset...'.format(dataset_str))
        labels = np.loadtxt("./data/snd/snd_true_idx.txt",dtype='int')
        Metric_label = labels
        labels = get_labelmatrix(labels)
        # labels = get_labelmatrix(labels)

        node_num = labels.shape[0]
        labels_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_vttdata(node_num)
        numLay = 3
        edge_index_total = {}
        for l in range(numLay):
            adj = np.genfromtxt("./data/snd/snd_Adj_Layer"+str(l+1)+".txt", dtype=np.int32)
            adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),
                                shape=(node_num, node_num),
                                dtype=np.float32)
            # adj = normalize(adj + sp.eye(adj.shape[0]))  # shape= （2708， 2708）， type = scipy.sparse.csr.csr_matrix
            adj = sparse_mx_to_torch_sparse_tensor(adj)  # shape =torch.Size([2708, 2708], type = torch.Tensor

            edge_index = np.loadtxt("./data/snd/snd_Adj_Layer"+str(l+1)+".txt")
            features = np.genfromtxt("./data/snd/snd_Feature_Layer"+str(l+1)+".txt")
            features = sp.csr_matrix(features, dtype=np.float32)  #type=scipy.sparse.csr.csr_matrix,
            features = normalize(features)
            features = torch.FloatTensor(
                np.array(features.todense()))
            Feature[l] = features
            print("features is.....{}".format(features.shape))
            Adj[l] = adj
            edge_index_total[l] = edge_index
        edge = edge_index_total[0]
        for lay in range(1, numLay):
            edge = np.vstack([edge, edge_index_total[lay]])
        edge = np.unique(edge, axis=0)

        Kmeans_F = 0
        G = 0
        kmeans_features_labels = 0
        return Adj, Feature, labels, kmeans_features_labels, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge



def load_data_our(dataset_str,dice, numLay, method = None):


    adj, features, labels, kmeans_features_labels, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge = load_multilayerNet(
            dataset_str, dice, numLay)
    print(numLay)

    return adj, features, labels, kmeans_features_labels, idx_train, idx_val, idx_test, G, numLay, Metric_label, edge

def load_data(dataset_str, dice): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str)) #
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':

        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    G = nx.from_scipy_sparse_matrix(adj)

    print("the type adj is :{}".format(type(adj)))
    labels = np.vstack((ally, ty))#

    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    I=np.eye(adj.shape[0])
    adj = adj + dice * I

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    labels_num = labels.shape[1]
    kmeans_features_labels = kmeans(labels_num, features)
    return adj, features, labels, kmeans_features_labels, idx_train, idx_val, idx_test, G

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation""" #
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def convert_label(label_matrix):
    label_num = label_matrix.shape[1]
    node_num = label_matrix.shape[0]
    label_assemble = []
    for l in range (label_num):
        label_assemble.append([])
    for i in range(node_num):
        for j in range(label_num):
            if label_matrix[i,j] == 1:
                label_assemble[j].append(i)

    return label_assemble

def convert_label2(labels):
    k = len(Counter(labels))
    label_assemble = []
    for l in range (k):
        label_assemble.append([])
    for i, element in enumerate(labels):
        label_assemble[element].append(i)
    return label_assemble


def modularity_generator(G):
    """
    Function to generate a modularity matrix.
    :param G: Graph object.
    :return laps: Modularity matrix.
    """
    print("Modularity calculation.\n")
    degrees = nx.degree(G)
    e_count = len(nx.edges(G))
    modu = np.array([[float(degrees[node_1]*degrees[node_2])/(2*e_count) for node_1 in nx.nodes(G)] for node_2 in tqdm(nx.nodes(G))],dtype = np.float64)
    return modu

def kmeans(k, embeddings):
    clf = KMeans(k)  # k-means聚类为7类

    y_pred = clf.fit_predict(embeddings)
    return y_pred

def get_vttdata(node_num):
    all=range(node_num)
    idxes = np.random.choice(all, int(node_num*0.6))
    idx_train, idx_val, idx_test = idxes[:int(node_num * 0.1)], idxes[int(node_num * 0.2):int(node_num * 0.4)], idxes[int(node_num * 0.4):int(node_num * 0.6)]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test

def get_adj_lilmatrix(edge_path, node_num):
    A = sp.lil_matrix((node_num, node_num), dtype=int)

    with open(edge_path, 'r') as fp:
        content_list = fp.readlines()

        for line in content_list[0:]:
            line_list = line.split("\t")


            from_id, to_id = line_list[0], line_list[1]
            # remove self-loop data
            if from_id == to_id:
                continue

            A[int(from_id), int(to_id)] = 1   #
            A[int(to_id), int(from_id)] = 1

    return A

def get_labelmatrix(labels):
    node_num = labels.shape[0]
    print("the shape of labels is:{}".format(labels.shape))
    labels = labels.tolist()
    labels_num = len(Counter(labels))
    labels_matrix = np.zeros(shape=(node_num,labels_num),dtype=int)
    print("the shape of label_matirx is:{0}".format(labels_matrix.shape))
    for i in range(node_num):
        labels_matrix[i,int(labels[i])] = 1
    return labels_matrix

def savegraph_edges(g):
    edges_list = g.edges()
    np.savetxt(".\pubmed\pubmed_edges.txt", edges_list, fmt="%d")
    print("saved_edges!!!!!")

def savegraph_labels(labels):
    labels_list = []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j]==1:
                labels_list.append(j)
    np.savetxt(".\pubmed\pubmed_labels.txt", labels_list, fmt="%d")
    print("saved_labels!!!!!")

def adj_Generate_new(edge_index, edge_logits, nodenum,drop_prob2):#它将edge_logits中的值赋给特征矩阵中的相应节点对位置，以建立节点之间的关系。
    feature = torch.zeros(nodenum, nodenum)
    row, _ = np.shape(edge_index)
    for line in range(row):
        if edge_logits[line, 0]<drop_prob2:
            feature[int(edge_index[line, 0]), int(edge_index[line, 1])]=0
        else:
            feature[int(edge_index[line, 0]), int(edge_index[line, 1])] = 1
    return feature


def feature_Generate_new(edge_index, edge_logits, nodenum,drop_prob2):
    feature = torch.zeros(nodenum, nodenum)
    row, _ = np.shape(edge_index)
    for line in range(row):
        if edge_logits[line, 0]<drop_prob2:
            feature[int(edge_index[line, 0]), int(edge_index[line, 1])]=0
        else:
            feature[int(edge_index[line, 0]), int(edge_index[line, 1])] = 1
    return feature


def adj_Generate_new2(adj,dice,node_num):
    A=adj.detach().numpy()
    sparse_adjacency_matrix = csr_matrix(A)
    g = nx.from_scipy_sparse_matrix(sparse_adjacency_matrix)
    I = np.eye(adj.shape[0])
    adj_new = adj.detach().numpy() + dice * I
    adj_new = normalize_adj(adj_new + sp.eye(adj_new.shape[0]))
    adj_new = sparse_mx_to_torch_sparse_tensor(adj_new)
    return adj_new

def feature_Generate_new2(edge_index, edge_logits, nodenum,drop_prob2):
    feature = torch.zeros(nodenum, nodenum)
    row, _ = np.shape(edge_index)
    for line in range(row):
        if edge_logits[line, 0]<drop_prob2:
            feature[int(edge_index[line, 0]), int(edge_index[line, 1])]=0
        else:
            feature[int(edge_index[line, 0]), int(edge_index[line, 1])] = 1
    return feature


def feature_Generate(edge_index, edge_logits, nodenum):
    feature = torch.zeros(nodenum, nodenum)
    row, _ = np.shape(edge_index)
    for line in range(row):
        feature[int(edge_index[line, 0]), int(edge_index[line, 1])] = edge_logits[line, 0]
    return feature


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx