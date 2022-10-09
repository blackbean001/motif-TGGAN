import os.path

import networkx as nx
import scipy.io
import scipy as sp
import io
import numpy as np

def save_mtx(data_path, num_nodes, src_cid, dst_cid):
    with open(data_path, 'r') as f:
        lines = f.readlines()

    G = nx.Graph()
    G.add_nodes_from(list(range(num_nodes)))
    for line in lines:
        item_list = line.strip().split(" ")
        src, dst = item_list[src_cid], item_list[dst_cid]
        G.add_edge(int(eval(src)), int(eval(dst)))

    a = nx.to_scipy_sparse_matrix(G)
    output_path = os.path.dirname(data_path)
    output_name = os.path.basename(data_path).split(".")[0] + ".mtx"
    sp.io.mmwrite(os.path.join(output_path, output_name), a, field="pattern")

    return os.path.join(output_path, output_name)
