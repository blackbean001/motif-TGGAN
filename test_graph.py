# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:55:46 2022

@author: MrBlackBean
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_graph(cut):
    
    fig, axs = plt.subplots(2, 1)
    
    fake_data_path = "/Users/MrBlackBean/Dropbox/lab_working_directory/algorithms/TGGAN_torch/fake_graphs.npy"
    fake_graphs = np.load(fake_data_path)
    G_fake = nx.MultiDiGraph()
    for i in fake_graphs[:cut]:
        G_fake.add_edge(int(i[1]), int(i[2]))
    nx.draw(G_fake, ax=axs[0])
    
    
    real_data_path = "/Users/MrBlackBean/Dropbox/lab_working_directory/algorithms/TGGAN_torch/data/auth/auth_train.txt"
    real_graphs = np.loadtxt(real_data_path)
    G_real = nx.MultiDiGraph()
    for i in real_graphs[:cut]:
        G_real.add_edge(int(i[1]), int(i[2]))
    nx.draw(G_real, ax=axs[1])
    
    return G_fake, G_real


def draw_degree_dist(G):
    # draw degree distribution
    
    degree_sequence = sorted((d for n, d in G.degree().items()), reverse=True)
    dmax = max(degree_sequence)
    
    fig, axs = plt.subplots(3, 1)    
    
    Gcc = G.subgraph(sorted(nx.weakly_connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc)
    nx.draw_networkx_nodes(Gcc, pos, ax=axs[0], node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=axs[0], alpha=0.4)
    
    ax1 = axs[1]
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    
    ax2 = axs[2]
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")
    
    fig.tight_layout()
    plt.show()
    

# use semi-supervised learning to generate label
def add_label(ModelObject, model_path):
    model = torch.load(model_path)



if __name__ == "__main__":
    G_fake, G_real = load_graph(1000000)
    draw_degree_dist(G_real)
    draw_degree_dist(G_fake)




















