import os
import argparse
from data import *
import logging
import time
import numpy as np
import datetime
from evaluation import *
from tggan import *
from tqdm import tqdm
import torch
from torch import autograd
import torch.nn.functional as F
from get_motif import *

use_reward = True

#n_nodes = 27
n_nodes = 3784
rw_len = 8
gpu_id = 0
t_end = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The gpu we are using is: ", str(gpu_id))
torch.cuda.set_device(int(gpu_id))
batch_size = 32
noise_dimension = 128
noise_type = "Gaussian"
embedding_size = int(n_nodes // 2)
#continue_training = "auth-202209281704"
continue_training = "bitcoin_alpha-202210081724"

n_eval_loop = 40
n_samples = batch_size * 10
transitions_per_iter = batch_size * n_eval_loop
eval_transitions = transitions_per_iter * 10

def generate_discrete(generator, n_samples, n_eval_loop):
    generator.eval()

    start_x_0 = F.one_hot(torch.zeros(n_samples, 1).long(), 2)[:,0,:].to(device)
    start_x_1 = F.one_hot(torch.zeros(n_samples, 1).long(), 2)[:,0,:].to(device)
    start_t0 = torch.ones(n_samples, 1).float().to(device)
    #print("start_x_0: ", start_x_0)
    #print("start_t0: ", start_t0)
    fake_x, fake_t0, fake_e, fake_tau, fake_end = [], [], [], [], []

    for i in range(n_eval_loop):
        initial_states_noise = make_noise([n_samples, noise_dimension], noise_type).to(device)
        if i == 0:
            fake_x_output, fake_t0_res_output, \
            fake_node_outputs, fake_tau_outputs, \
            fake_end_output = generator(initial_states_noise, x_input=start_x_1.float(), t0_input=start_t0)
        else:
            if rw_len == 1:
                t0_input = fake_tau_outputs[:, -1, :]
                fake_x_output, fake_t0_res_output, \
                fake_node_outputs, fake_tau_outputs, \
                fake_end_output = generator(initial_states_noise, x_input=start_x_0, t0_input=t0_input)
            else:
                t0_input = fake_tau_outputs[:, 0, :]
                edge_input = fake_node_outputs[:, 2:, :]
                tau_input = fake_tau_outputs[:, 1:, :]
                fake_x_output, fake_t0_res_output, \
                fake_node_outputs, fake_tau_outputs, \
                fake_end_output = generator(initial_states_noise, x_input=start_x_0, t0_input=t0_input, \
                                            edge_input=edge_input, tau_input=tau_input)

        fake_x_outputs_discrete = torch.argmax(fake_x_output, -1)
        fake_node_outputs_discrete = torch.argmax(fake_node_outputs, -1)
        fake_end_discretes = torch.argmax(fake_end_output, -1)

        fake_x.append(fake_x_outputs_discrete)
        fake_t0.append(fake_t0_res_output)
        fake_e.append(fake_node_outputs_discrete)
        fake_tau.append(fake_tau_outputs)
        fake_end.append(fake_end_discretes)

    return fake_x, fake_t0, fake_e, fake_tau, fake_end

generator = Generator(n_nodes, rw_len,
                      t_end, device, batch_size, noise_dimension,
                      noise_type,
                      use_gumbel=True,
                      use_decoder='normal',
                      generator_x_up_layers=[64],
                      generator_t0_up_layers=[128],
                      generator_tau_up_layers=[128],
                      constraint_method='min_max',
                      generator_layers=[40],
                      W_down_generator_size=embedding_size)

resume_path = "output/{}/save_models/model-snapshots-generator.pt".format(continue_training)
#generator.load_stat_dict(torch.load(os.path.join(resume_path, "generator.pth")))
generator.load_state_dict(torch.load(resume_path))
generator.to(device)

print("start generating graphs")
generator.eval()
n_eval_iters = int(eval_transitions / n_samples)  # =n_eval_loop
fake_graphs = []
fake_x_t0 = []
print("n_eval_iters, n_eval_loop: {}, {}".format(n_eval_iters, n_eval_loop))
for q in range(n_eval_iters):
    fake_x, fake_t0, fake_edges, fake_t, fake_end = \
        generate_discrete(generator, n_samples, n_eval_loop)
    node_logit = generator.logit
    smpls = None
    stop = [False] * n_samples
    for i in range(n_eval_loop):
        x, t0, e, tau, le = fake_x[i], fake_t0[i], fake_edges[i], fake_t[i], fake_end[i]
        # print("x shape: ", x.shape)     # (bs*10)
        # print("t0 shape: ", t0.shape)   # (bs*10, 1)
        # print("e shape: ", e.shape)     # (bs*10, rw_len*2)
        # print("tau shape: ", tau.shape) # (bs*10, rw_len, 1)
        # print("le shape: ", le.shape)   # (bs*10)
        if q == 0 and i >= n_eval_loop - 3:
            print('eval_iters: {} eval_loop: {}'.format(q, i))
            print('eval node logit min: {} max: {}'.format(node_logit.min(), node_logit.max()))
            print('generated [x, t0, e, tau, end]\n[{}, {}, {}, {}, {}]'.format(
                x[0], t0[0, 0], e[0, :], tau[0, :, 0], le[0]
            ))
            print('generated [x, t0, e, tau, end]\n[{}, {}, {}, {}, {}]'.format(
                x[1], t0[1, 0], e[1, :], tau[1, :, 0], le[1]
            ))
        e = e.reshape(-1, rw_len, 2).cpu().numpy()
        tau = tau.reshape(-1, rw_len, 1).cpu().numpy()
        if i == 0:
            smpls = np.concatenate([e, tau], axis=-1)
            # print("smpls0: ", smpls.shape) # (2560, 4, 3)
        else:
            new_pred = np.concatenate([e[:, -1:], tau[:, -1:]], axis=-1)
            # print("new_pred: ", new_pred.shape) # (2560, 1, 3)
            smpls = np.concatenate([smpls, new_pred], axis=1)
            # print("smpls1: ", smpls.shape) # (2560, n_eval_loop+3, 3)

        # judge if reach max length
        for b in range(n_samples):
            b_le = le[b]

            if i == 0 and b_le == 1:  # end
                stop[b] = True
            if i > 0 and stop[b]:  # end
                smpls[b, -1, :] = -1
            if i > 0 and not stop[b] and b_le == 1:
                stop[b] = True
    fake_x = np.array([x.cpu().numpy() for x in fake_x]).reshape(-1, 1)
    fake_t0 = np.array([x.cpu().numpy() for x in fake_t0]).reshape(-1, 1)
    fake_len = np.array([x.cpu().numpy() for x in fake_end]).reshape(-1, 1)  # change to end
    fake_start = np.c_[fake_x, fake_t0, fake_len]
    fake_x_t0.append(fake_start)
    fake_graphs.append(smpls)

# reformat fake_graph and remove -1 value
fake_graphs = np.array(fake_graphs)
fake_graphs = convert_graphs(fake_graphs)
fake_graphs[:, 3] = t_end - fake_graphs[:, 3]
np.save("fake_graphs.npy", fake_graphs)


