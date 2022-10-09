# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:33:21 2022

@author: MrBlackBean
"""


import numpy as np

class TemporalWalkDataset:
    """
    Helper class to generate temporal random walks on the input user-trips matrix.
    The matrix gets shape: [day, hour, origin, destination]
    Parameters
    -----------
    edges: edges [[d, i, j]], shape: samples x 3
    edges_times: real time of edges [time], shape: samples x 1
    """

    def __init__(self, edges, t_end,
                 rw_len=4, init_walk_method='uniform', batch_size = 56):
  #      if edges.shape[1] != 4: raise Exception('edges must have shape: samples x 4')

        self.t_end = t_end
        self.edges_days = edges[:, [0]]     
        self.edges = edges[:, [1, 2]]
        self.edges_times = edges[:, [3]]
        self.rw_len = rw_len
        self.init_walk_method = init_walk_method
        self.batch_size = batch_size
        self.total_batches = int(len(self.edges) / batch_size)
        #print("total number of batches: {}".format(int(len(self.edges) / self.rw_len / batch_size)))
        print("total number of batches: {}".format(int(len(self.edges) / batch_size)))

    def __getitem__(self, index):
        return temporal_random_walk(
            self.edges_days, self.edges, self.edges_times, self.t_end,
            self.rw_len, self.init_walk_method)
    
    def __len__(self):
        #return int(len(self.edges) / self.rw_len)
        return int(len(self.edges))
# @jit(nopython=True)
def temporal_random_walk(edges_days, edges, edges_times, t_end,
                         rw_len,init_walk_method):

    unique_days = np.unique(edges_days.reshape(1, -1)[0])

    while True:
        # select a day with uniform distribution
        walk_day = np.random.choice(unique_days)
        mask = edges_days.reshape(1, -1)[0] == walk_day
        # subset for this day
        walk_day_edges = edges[mask]
        walk_day_times = edges_times[mask]
        # select a start edge. and unbiased or biased to the starting edges
        n = walk_day_edges.shape[0]
        if n >= rw_len: break

    n = n - rw_len + 1
    if init_walk_method == 'uniform': probs = Uniform_Prob(n)
    elif init_walk_method == 'linear': probs = Linear_Prob(n)
    elif init_walk_method == 'exp': probs = Exp_Prob(n)
    else: raise Exception('wrong init_walk_method!')
    
    # get start index
    if n == 1: start_walk_inx = 0
    else: start_walk_inx = np.random.choice(n, p=probs)            
    
    selected_walks = walk_day_edges[start_walk_inx:start_walk_inx + rw_len] # πÃ∂®≥§∂»
    selected_times = walk_day_times[start_walk_inx:start_walk_inx + rw_len]

    # get start residual time
    if start_walk_inx == 0: t_res_0 = t_end
    else:
        # print('selected start:', selected_walks[0])
        t_res_0 = t_end - walk_day_times[start_walk_inx-1, 0]

    # convert to residual time  £”‡ ±º‰
    selected_times = t_end - selected_times

    # add a stop sign of -1
    x = 1
    if start_walk_inx > 0: x = 0
    walks_mat = np.c_[selected_walks, selected_times]
    if rw_len > len(selected_walks):
        n_stops = rw_len - len(selected_walks)
        walks_mat = np.r_[walks_mat, [[-1, -1, -1]] * n_stops]

    # add start resdidual time
    if start_walk_inx == n-1:
        is_end = 1.
    else:
        is_end = 0.
    walks_mat = np.r_[[[x] + [is_end] + [t_res_0]], walks_mat]
    
    return np.array(walks_mat)  # walks_mat: (rw_len+1, 3)
    
def Exp_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = 1. / np.arange(1, n + 1, dtype=np.int)
    #     c = np.cbrt(1. / np.arange(1, n+1, dtype=np.int))
    exp_c = np.exp(c)
    return exp_c / exp_c.sum()

def Linear_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = np.arange(n+1, 1, dtype=np.int)
    return c / c.sum()

def Uniform_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = [1./n]
    return c * n

def Split_Train_Test(edges, train_ratio):
    days = sorted(np.unique(edges[:, 0]))
    if len(days) != 1:
        b = days[int(train_ratio * len(days))]
        train_mask = edges[:, 0] <= b
    else:
        train_mask = np.zeros_like(edges[:, 0])
        train_mask[:int(len(edges) * train_ratio)] = 1
    
    train_mask = train_mask.astype(int)
    train_edges = edges[train_mask][:,:4]
    test_edges = edges[train_mask==0][:,:4]
       
    return train_edges, test_edges
    

    
    
    
    
    
    
