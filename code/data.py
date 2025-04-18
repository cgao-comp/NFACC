import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn import preprocessing
import math
import networkx as nx
from utils.dice import getJaccard_similarity
from utils.process import get_adj_lilmatrix
from torch.autograd import Variable
from sklearn.cluster import KMeans
from sklearn import metrics
from layers import GCN, AvgReadout, Discriminator
from models import MLP
from utils.process import sparse_mx_to_torch_sparse_tensor
import functools
import operator
import collections
import pandas as pd
import torch.nn.functional as F

from models import DGI_node2, gcn_network2, DGI, DGI1, DGI_network, DGI_node, gcn_network, LogReg  # 0312
from utils import process, clustering, RDropLoss
from utils.clustering import convertMatrix_listlabel


import networkx as nx


def read_edge_list(file_path):

    with open(file_path, 'r') as file:
        edges = [tuple(map(int, line.strip().split())) for line in file]
    return edges


def graph_basic_info(edge_list):

    G = nx.Graph()
    G.add_edges_from(edge_list)

    # 计算基础信息
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    clustering_coefficient = nx.average_clustering(G)

    # 扩充信息
    diameter = nx.diameter(G) if nx.is_connected(G) else "Graph is disconnected"
    num_connected_components = nx.number_connected_components(G)

    return {
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Average Degree": avg_degree,
        "Clustering Coefficient": clustering_coefficient,
        "Diameter": diameter,
        "Number of Connected Components": num_connected_components
    }



dataset_list=['9L128/128-0.5-0.2-9','9L128/128-0.5-0.3-9','9L128/128-0.5-0.4-9','9L128/128-0.5-0.5-9','9L128/128-0.5-0.6-9','9L128/128-0.5-0.7-9','9L128/128-0.5-0.8-9']
# dataset_list=['9L128/128-0.6-0.2-9','9L128/128-0.6-0.3-9','9L128/128-0.6-0.4-9','9L128/128-0.6-0.5-9','9L128/128-0.6-0.6-9','9L128/128-0.6-0.7-9','9L128/128-0.6-0.8-9']
for da in range(7):
    dataset=dataset_list[da]
    laynum=9
    print(dataset)
    max_e=0
    min_e=10000000
    for i in range(laynum):
        print(f"layer: {i+1}")
        file_path = f"./data/{dataset}_Adj_Layer{i+1}.txt"
        edge_list = read_edge_list(file_path)
        info = graph_basic_info(edge_list)

        # 输出结果
        print("Graph Basic Information:")
        for key, value in info.items():
            print(f"{key}: {value}")