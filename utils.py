# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:07:59 2022

@author: MrBlackBean
"""

import torch
import numpy as np

def make_noise(shape, type="Gaussian", z=None):
    if z == None:
        if type == "Gaussian":
            noise = torch.randn(shape[0], shape[1])
        elif type == 'Uniform':
            noise = torch.rand(shape[0], shape[1])
        else:
            raise AssertionError("ERROR: Noise type {} not supported".format(type))
    else:
        noise = z
    return noise
    
    
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
    
    
    
def time_constraint(t, epsilon=1e-1, method='min_max'):
    if method == 'relu':
        t = nn.ReLU(t) - nn.ReLU(t - 1.)
    elif method == 'clip':
        t = torch.clamp(t, 0., 1.)
    elif method == 'min_max':
        min_t = torch.min(t)
        max_t = torch.max(t)
        if max_t == min_t:
            min_max = torch.ones_like(t)
        else:
            min_max = (t - min_t) / (max_t - min_t)
    return min_max
    
def get_num_nodes(edges, start_from_0=True):
    num_nodes = 0    
    for edge in edges:
        num_nodes = max(num_nodes, max(edge[0], edge[1]))
    if start_from_0 == True:
        return int(num_nodes+1)
    if start_from_0 == False:
        return int(num_nodes)
    
def convert_graphs(fake_graphs):
    print("fake_graph1.shape: ", fake_graphs.shape)  #  (40, 2560, 43, 3)
    _, _, e, k = fake_graphs.shape  # (n_eval_loop, bs*10, n_eval_loop+3, 3)
    fake_graphs = fake_graphs.reshape([-1, e, k])
    print("fake_graph2.shape: ", fake_graphs.shape) # (102400, 43, 3)
    tmp_list = None
    for d in range(fake_graphs.shape[0]):
        d_graph = fake_graphs[d]
        d_graph = d_graph[d_graph[:, 2] > 0.]
        d_graph = np.c_[np.array([[d]] * d_graph.shape[0]), d_graph]
        if tmp_list is None:
            tmp_list = d_graph
        else:
            tmp_list = np.r_[tmp_list, d_graph]
    return tmp_list
