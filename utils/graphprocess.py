import numpy as np


def floyd_dis(adj_mx):  #计算距离意义上的最短路径
    num_of_vertices=adj_mx.shape[0]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.float32)

    path = -1*np.ones((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.int16)
    
    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            A[i,j]=adj_mx[i,j]

    for k in range(0, num_of_vertices):
        for i in range(0, num_of_vertices):
            if np.isinf(A[i,k]):
                continue
            for j in range(0, num_of_vertices):
                if np.isinf(A[k,j]):
                    continue
                else:
                    if np.isinf(A[i,j]) or A[i,j]>A[i,k]+A[k,j]:
                        A[i,j]=A[i,k]+A[k,j]
                        path[i,j]=k

    return A,path

def floyd_hop(adj_mx):   #计算跳数意义上的最短路径
    num_of_vertices=adj_mx.shape[0]

    path_len = -1*np.ones((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.int16)
    
    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            if i==j:
                path_len[i,j]=0
            if ~np.isinf(adj_mx[i,j]):
                path_len[i,j]=1

    for k in range(0, num_of_vertices):
        for i in range(0, num_of_vertices):
            if path_len[i,k]==-1:
                continue
            for j in range(0, num_of_vertices):
                if path_len[k,j]==-1:
                    continue
                else:
                    if path_len[i,j]==-1 or path_len[i,j]>path_len[i,k]+path_len[k,j]:
                        path_len[i,j]=path_len[i,k]+path_len[k,j]

    return path_len

def gen_attention_bias(supports):
    shortest_path_result,_=floyd_dis(supports)
    shortest_hop_result=floyd_hop(supports)

    # Calculates the standard deviation as theta.
    distances = shortest_path_result[~np.isinf(shortest_path_result)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(shortest_path_result / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < 0.1] = 0

    shortest_hop_result+=1

    return adj_mx,shortest_hop_result

def gen_node_bias(adj_mx):
    num_of_vertices=adj_mx.shape[0]
    in_degree = -1*np.ones((int(num_of_vertices)),
                        dtype=np.int16)

    out_degree= -1*np.ones((int(num_of_vertices)),
                        dtype=np.int16)

    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            if ~np.isinf(adj_mx[i,j]):
                in_degree[i]+=1
                out_degree[j]+=1
            
    return in_degree,out_degree

if __name__== "__main__" :
    from utils import load_pickle
    import os

    _, _, adj_mx = load_pickle(os.path.join('data', 'sensor_graph', 'adj_mat.pkl'))
