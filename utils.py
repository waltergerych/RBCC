import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import copy
from scipy.io import arff
import torch
import numpy as np
import scipy

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h

def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.DiGraph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    elif graph_layout == 'circular':
        graph_pos=nx.circular_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color, arrows=True, arrowsize=50)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
#     nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
#                                  label_pos=edge_text_pos)

    # show graph
    plt.show()
    
def class_index(label, classes):
    indx = [i for i,val in enumerate(classes) if val==label]
    return indx[0]
    
def load_scene(partition='Train'):
    if partition == 'Train':
        features = ['Att' + str(i) for i in range(1,295)]
        labels = ['Beach','Sunset','FallFoliage','Field','Mountain','Urban']
        data = arff.loadarff('datasets/scene-train.arff')
        df = pd.DataFrame(data[0])
        train_data = df[features]
        train_labels = df[labels].astype(float)
        return train_data, train_labels
    if partition == 'Test':
        features = ['Att' + str(i) for i in range(1,295)]
        labels = ['Beach','Sunset','FallFoliage','Field','Mountain','Urban']
        data = arff.loadarff('datasets/scene-test.arff')
        df = pd.DataFrame(data[0])
        test_data = df[features]
        test_labels = df[labels].astype(float)
        return test_data, test_labels
    

    
def make_binary(label_df, input_labels):
    label_df = copy.copy(label_df)[input_labels]
    for label in input_labels:
        label_df.loc[label_df[label] < 1, label] = 0
    return label_df

def get_parents_indxs(parent_list, classes):
    parents_indxs = []
    for p in parent_list:
        parents_indxs.append(class_index(p, classes))
    return parents_indxs

def get_parent_dict(adj_mat, classes):
    parent_dict = {}
    for i,c in enumerate(classes):
        d = adj_mat[c]
        parent_list = list(d[d==True].index)
        parents_indxs = get_parents_indxs(parent_list, classes)
        parent_dict[i] = {}
        parent_dict[i]['class'] = c
        parent_dict[i]['parents'] = parents_indxs
    return parent_dict

