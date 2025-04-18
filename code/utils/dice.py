import networkx as nx
import scipy.sparse as sp
import numpy as np
from sklearn import preprocessing

def getLs(orbitFile_url):
    GDVM = np.loadtxt(orbitFile_url)
    S=cos_similarity(GDVM)
    Ls=Laplacian(S)
    return S

def cos_similarity(array):
    n=array.shape[0]

    print('begin standardization')
    array=preprocessing.scale(array)

    vector_norm=np.linalg.norm(array, axis=1)
    S=np.zeros((n,n))

    for i in range(n):

        S[i,i]=1
        for j in range(i+1,n):
            #if W[i,j]!=0:
            S[i,j]= np.dot( array[i,:],array[j,:] ) / (vector_norm[i]*vector_norm[j])
            S[i,j]=0.5+0.5*S[i,j]
            S[j,i]= S[i,j]
    return S

def Laplacian(W):

    d=[np.sum(row) for row in W]
    D=np.diag(d)

    Dn=np.ma.power(np.linalg.matrix_power(D,-1),0.5)
    Lbar=np.dot(np.dot(Dn,W),Dn)

    return np.mat(Lbar)


def getGraph(edges_path):

    G = nx.Graph()
    with open(edges_path, 'r') as fp:
        content_list = fp.readlines()

        for line in content_list[0:]:
            line_list = line.split(" ")
            from_id, to_id = line_list[0], line_list[1]

            if from_id == to_id:
                continue
            G.add_edge(int(from_id),int(to_id))

    return G

def getSimilariy_modified(node_num, graph):

    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    edges_list = list(graph.edges())
    node_list = list(graph.node())
    for i, node in enumerate(node_list):

        neibor_i_list = list(graph.neighbors(node))
        first_neighbor = neibor_i_list

        for k, second_nighbor in enumerate(first_neighbor):
            second_list = list(graph.neighbors(second_nighbor))
            neibor_i_list = list(set(neibor_i_list).union(set(second_list)))

        neibor_i_num = len(first_neighbor)
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors(node_j))
            neibor_j_num = len(neibor_j_list)
            commonNeighbor_list = [x for x in first_neighbor if x in neibor_j_list]

            commonNeighbor_num = len(commonNeighbor_list)
            neibor_i_num_x = neibor_i_num
            if (i,j) in edges_list:
                commonNeighbor_num = commonNeighbor_num + 2
                neibor_j_num = neibor_j_num + 1
                neibor_i_num_x = neibor_i_num + 1

            similar_matrix[node, node_j] = (2*commonNeighbor_num)/(neibor_j_num + neibor_i_num_x)
    return similar_matrix


def getJaccard_similarity(node_num, graph):

    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    edges_list = list(graph.edges())
    node_list = list(graph.nodes())
    for i, node in enumerate(node_list):

        neibor_i_list = list(graph.neighbors(node))
        first_neighbor = neibor_i_list

        for k, second_nighbor in enumerate(first_neighbor):
            second_list = list(graph.neighbors(second_nighbor))
            neibor_i_list = list(set(neibor_i_list).union(set(second_list)))

        neibor_i_num = len(first_neighbor)
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors(node_j))
            fenzi = len(list(set(first_neighbor).intersection(set(neibor_j_list))))
            fenmu = len(list(set(first_neighbor).union(set(neibor_j_list))))
            similar_matrix[node, node_j] = fenzi / fenmu
    return similar_matrix


def dice_similarity_matrix(adj_matrix):

    n_nodes = adj_matrix.shape[0]
    dice_sim = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                dice_sim[i, j] = 1.0
            else:
                intersection = np.sum(np.logical_and(adj_matrix[i], adj_matrix[j]))
                sum_sizes = np.sum(adj_matrix[i]) + np.sum(adj_matrix[j])
                dice_sim[i, j] = 2 * intersection / sum_sizes if sum_sizes > 0 else 0

    return dice_sim