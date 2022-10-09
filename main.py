# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:02:27 2022

@author: Administrator
"""

import configparser
import os
import argparse
from data import *
import logging
import time
import numpy as np
import datetime
from evaluation import *
from tggan import *
from utils import *
from tqdm import tqdm
import torch
from torch import autograd
import torch.nn.functional as F
from get_motif import *

torch.backends.cudnn.enabled = False
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description="TGGAN parser")

parser.add_argument("-conf", "--config", default="config.txt", type=str)
parser.add_argument("-bs", "--batch_size", default=64, type=int)
parser.add_argument("-tr", "--train_ratio", default=0.8, type=float)
parser.add_argument("-te", "--t_end", default=1, type=int)
parser.add_argument("-gid", "--gpu_id", default=0, type=int)
parser.add_argument("-ep", "--epochs", default=10000, type=int)
parser.add_argument("-lr", "--learningrate", default=0.0003, type=float,
                    help="if this run should run all evaluations")
parser.add_argument("-rl", "--rw_len", default=8, type=int,
                    help="random walks maximum length in DeepTemporalWalk")
parser.add_argument("-uw", "--use_wgan", default=True, type=bool,
                    help="if use WGAN loss function")
parser.add_argument("-ud", "--use_decoder", default='deep', type=str,
                    help="if decoder function")
parser.add_argument("-cm", "--constraint_method", default='min_max', type=str,
                    help="time constraint computing method")
parser.add_argument("--early_stopping", default=0.0001, type=float,
                    help="stop training if evaluation metrics are good enough")
parser.add_argument("-ct", "--continueTraining", default='', type=str,
                    help="if this run is restored from a corrupted run")
parser.add_argument("-iw", "--init_walk_method", default='uniform', type=str,
                    help="TemporalWalk sampler")
parser.add_argument("-nd", "--noise_dimension", default=128, type=int)
parser.add_argument("-nt", "--noise_type", default="Gaussian", type=str)
parser.add_argument("-ev", "--eval_every", default=10, type=int, \
                    help="evaluation interval of epochs")
parser.add_argument("-di", "--discriminator_iter", default=3, type=int, \
                    help="discriminator iterations")
parser.add_argument("-ec", "--edge_contact_time", default=0.01, type=float,
                    help="stop training if evaluation metrics are good enough")
parser.add_argument("-ne", "--n_eval_loop", default=40, type=int,
                    help="number of walk loops")
parser.add_argument("-rv", "--reward_every", default=1, type=int, \
                    help="evaluation interval of epochs")
parser.add_argument("-ur", "--use_reward", default=True, type=bool, \
                    help="evaluation interval of epochs")
parser.add_argument("-rp", "--reward_pretraining", default=20, type=int)

# parse arguments
args = parser.parse_args()
configFilePath = args.config

reward_pretraining = args.reward_pretraining
train_ratio = args.train_ratio
t_end = args.t_end
gpu_id = args.gpu_id
lr = args.learningrate
epochs = args.epochs
continue_training = args.continueTraining
use_wgan = args.use_wgan
use_decoder = args.use_decoder
constraint_method = args.constraint_method
rw_len = args.rw_len
batch_size = args.batch_size
init_walk_method = args.init_walk_method
noise_dimension = args.noise_dimension
noise_type = args.noise_type
eval_every = args.eval_every
reward_every = args.reward_every
discriminator_iter = args.discriminator_iter
edge_contact_time = args.edge_contact_time
n_eval_loop = args.n_eval_loop
transitions_per_iter = batch_size * n_eval_loop
eval_transitions = transitions_per_iter * 10
early_stopping = args.early_stopping
use_reward = args.use_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The gpu we are using is: ", str(gpu_id))
torch.cuda.set_device(int(gpu_id))

def process_input(real_data, N):
    real_edge_inputs_discrete = real_data[:, 1:, 0:2].to(torch.long)
    real_node_inputs_discrete = torch.reshape(real_edge_inputs_discrete,
                                              (batch_size, rw_len * 2))

    real_node_inputs = F.one_hot(real_node_inputs_discrete, N)  # (bs, rw*2, N)
    real_tau_inputs = real_data[:, 1:, 2:3]  # (bs, rw)
    real_x_input_discretes = real_data[:, 0, 0].to(torch.long)
    real_x_inputs = F.one_hot(real_x_input_discretes, 2)  # (bs, 2)
    real_end_discretes = real_data[:, 0, 1].to(torch.long)
    real_ends = F.one_hot(real_end_discretes, 2)  # (bs, 2)
    real_t0_res_inputs = real_data[:, 0:1, 2]  # (bs, 1)

    #return real_node_inputs, real_tau_inputs, real_x_inputs, \
    #        real_ends, real_t0_res_input

    return torch.tensor(real_node_inputs), torch.tensor(real_tau_inputs), \
            torch.tensor(real_x_inputs), torch.tensor(real_ends), torch.tensor(real_t0_res_inputs)

def interpolate_data(real_data, fake_data, name):
    batch_size = real_data.size(0)
    if name == 'node' or name == 'tau':
        eps = torch.rand(batch_size, 1, 1).to(real_data.device)
    elif name == 'x' or name =='end' or name == 't0_res':
        eps = torch.rand(batch_size, 1).to(real_data.device)
    #print(name, real_data.size(), fake_data.size(), eps.size())
    #node: torch.Size([32, 8, 593]) torch.Size([32, 8, 593]) torch.Size([32, 1, 1])
    #tau: torch.Size([32, 4, 1]) torch.Size([32, 4, 1]) torch.Size([32, 1, 1])
    #x: torch.Size([32, 2]) torch.Size([32, 2]) torch.Size([32, 1])
    #end: torch.Size([32, 2]) torch.Size([32, 2]) torch.Size([32, 1])
    #t0_res: torch.Size([32, 1]) torch.Size([32, 1]) torch.Size([32, 1])
    eps = eps.expand_as(real_data)
    eps.requires_grad = True
    interpolation = eps * real_data + (1 - eps) * fake_data

    return interpolation

def compute_gp(discriminator, real_node_inputs, real_tau_inputs, real_x_inputs, \
               real_ends, real_t0_res_inputs, fake_x_inputs, fake_t0_res_inputs, \
               fake_node_inputs, fake_tau_inputs, fake_ends):
    # interpolate data
    #print(real_node_inputs.size(), fake_node_inputs.size()) # (bs, 2*rw_len,num_nodes)
    #print(real_tau_inputs.size(), fake_tau_inputs.size())  # (bs, rw_len, 1)
    #print(real_x_inputs.size(), fake_x_inputs.size())  # (bs, 2)
    #print(real_ends.size(), fake_ends.size())  # (bs ,2)

    inter_node_inputs = interpolate_data(real_node_inputs, fake_node_inputs, "node")
    inter_tau_inputs = interpolate_data(real_tau_inputs, fake_tau_inputs, "tau")
    inter_x_inputs = interpolate_data(real_x_inputs, fake_x_inputs, "x")
    inter_ends = interpolate_data(real_ends, fake_ends, "end")
    inter_t0_res_inputs = interpolate_data(real_t0_res_inputs, fake_t0_res_inputs, "t0_res")

    inter_x_inputs = Variable(inter_x_inputs, requires_grad=True)
    inter_node_inputs = Variable(inter_node_inputs, requires_grad=True)
    inter_tau_inputs = Variable(inter_tau_inputs, requires_grad=True)
    inter_ends = Variable(inter_ends, requires_grad=True)
    inter_t0_res_inputs = Variable(inter_t0_res_inputs, requires_grad=True)

    # get logits
    #with torch.backends.cudnn.flags(enabled=False):
    interp_logits = discriminator(inter_x_inputs, inter_t0_res_inputs, \
                                      inter_node_inputs, inter_tau_inputs, inter_ends)
    grad_outputs = torch.ones_like(interp_logits)

    # compute gradients
    gradients = autograd.grad(outputs=interp_logits, \
                              inputs=(inter_x_inputs, inter_t0_res_inputs, \
                                      inter_node_inputs, inter_tau_inputs, inter_ends),
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True)[0]

    #gradients = autograd.grad(outputs=interp_logits, \
    #                          inputs=(inter_tau_inputs),
    #                          grad_outputs=grad_outputs,
    #                          create_graph=True,
    #                          retain_graph=True)[0]

    # compute and return gradient norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)

    return torch.mean((grad_norm - 1) ** 2)


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


def get_real_score(num_nodes, normalize=True):
    data_path = config.get("DATA", "datapath")

    # use pgd to get motifs
    output_file = save_mtx(data_path, num_nodes, 1, 2)
    pgd_output = output_file.split(".")[0] + ".macro"
    os.system("/home/lisong/algorithms/MTSN/PGD/pgd -f {} --macro {}".format(output_file, pgd_output))

    # parse pgd results
    result = []
    with open(pgd_output, 'r') as f:
        lines = f.readlines()
    for line in lines:
        result.append(line.replace(" ", "").split("=")[1])
    assert len(result) == 17

    if normalize:
        result = [int(r)/num_nodes for r in result]
    print("number of motifs in the real data: ", result)

    result = np.array(result).reshape(1, 17)

    return result


def run(config, args):
    dataname = config.get("DATA", "dataname")
    datapath = config.get("DATA", "datapath")

    dt = datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_name = '{}-{}'.format(dataname, dt)
    try:
        os.mkdir("output/{}".format(model_name))
        os.mkdir("output/{}/save_models".format(model_name))
        os.mkdir("output/{}/logs".format(model_name))
    except:
        1
    save_model_path = "output/{}/save_models/bestmodel-".format(model_name)

    logging.basicConfig(filename='output/{}/logs/{}'.format(model_name,model_name + '.log'))
    def log(str): print(str)

    print('use {} data'.format(dataname))

    edges = np.loadtxt(datapath)
    Edges = edges[:, 1:4]
    Edges = Edges.tolist()
    n_nodes = get_num_nodes(Edges, start_from_0=True)
    print("number of nodes: {}".format(n_nodes))
    embedding_size = int(n_nodes // 2)

    if not os.path.exists('data/{}/{}_train.txt'.format(dataname, dataname)):
        train_edges, test_edges = Split_Train_Test(edges, train_ratio)
        np.savetxt(fname='data/{}/{}_train.txt'.format(dataname, dataname), X=train_edges)
        np.savetxt(fname='data/{}/{}_test.txt'.format(dataname, dataname), X=test_edges)
    else:
        train_edges = np.loadtxt('data/{}/{}_train.txt'.format(dataname, dataname))
        test_edges = np.loadtxt('data/{}/{}_train.txt'.format(dataname, dataname))

    trainset = TemporalWalkDataset(train_edges, t_end, rw_len, init_walk_method, batch_size)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size)

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

    discriminator = Discriminator(n_nodes, rw_len,
                                  discriminator_layers=[30],
                                  W_down_discriminator_size=embedding_size)

    if use_reward:
        loss_reward = torch.nn.MSELoss()
        real_score = get_real_score(n_nodes, normalize=True)
        reward = RewardNet(n_nodes, rw_len, reward_every,
                           reward_layers=[15],
                           W_down_reward_size=embedding_size,
                           num_motif=17, real_score=torch.from_numpy(real_score))

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    if use_reward:
        reward = reward.to(device)

    if continue_training != '':
        resume_path = "output/{}/save_models/model-snapshots-".format(continue_training)
        generator.load_stat_dict(torch.load(resume_path, "generator.pth"))
        discriminator.load_stat_dict(torch.load(resume_path, "discriminator.pth"))
        if use_reward:
            reward.load_stat_dict(torch.load(resume_path, "reward.pth"))

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)
    if use_reward:
        optimizer_R = torch.optim.Adam(reward.parameters(), lr=lr, weight_decay=1e-5)

    mmd_avg_degree_min = 1000000
    if use_reward:
        record_generated_network = []
    for epoch in range(1, epochs):
        # update D network
        print("start epoch {}: ".format(epoch))
        for p in generator.parameters():
            p.requires_grad = False
        if use_reward:
            for p in reward.parameters():
                p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = True
        pbar = tqdm(total=trainset.total_batches)
        for i, batch in enumerate(train_loader):
            pbar.update(1)
            if batch.size()[0] != batch_size:
                continue

            batch = batch.to(device)
            real_node_inputs, real_tau_inputs, real_x_inputs, \
            real_ends, real_t0_res_inputs = process_input(batch, n_nodes)

            real_node_inputs = Variable(real_node_inputs.float(),requires_grad=True)
            real_tau_inputs = Variable(real_tau_inputs.float(),requires_grad=True)
            real_x_inputs = Variable(real_x_inputs.float(),requires_grad=True)
            real_ends = Variable(real_ends.float(),requires_grad=True)
            real_t0_res_inputs = Variable(real_t0_res_inputs.float(),requires_grad=True)

            # multiple run of discriminator
            for j in range(discriminator_iter):
                optimizer_D.zero_grad()

                initial_states_noise = make_noise([batch_size, noise_dimension], noise_type)
                initial_states_noise.requires_grad = True

                fake_x_inputs, fake_t0_res_inputs, \
                fake_node_inputs, fake_tau_inputs, \
                fake_ends = generator(initial_states_noise)

                fake_x_inputs.requires_grad = True
                fake_t0_res_inputs.requires_grad = True
                fake_node_inputs.requires_grad = True
                fake_tau_inputs.requires_grad = True
                fake_ends.requires_grad = True

                np.save("fake_x_inputs.npy", fake_x_inputs.data.cpu().numpy())
                np.save("fake_t0_res_inputs.npy", fake_t0_res_inputs.data.cpu().numpy())
                np.save("fake_node_inputs.npy", fake_node_inputs.data.cpu().numpy())
                np.save("fake_tau_inputs.npy", fake_tau_inputs.data.cpu().numpy())

                np.save("real_x_inputs.npy", real_x_inputs.data.cpu().numpy())
                np.save("real_t0_res_inputs.npy", real_t0_res_inputs.data.cpu().numpy())
                np.save("real_node_inputs.npy", real_node_inputs.data.cpu().numpy())
                np.save("real_tau_inputs.npy", real_tau_inputs.data.cpu().numpy())
                # process real data
                disc_real = discriminator(real_x_inputs, real_t0_res_inputs,
                                          real_node_inputs, real_tau_inputs, real_ends)
                disc_fake = discriminator(fake_x_inputs.detach(), fake_t0_res_inputs.detach(),
                                          fake_node_inputs.detach(), fake_tau_inputs.detach(), fake_ends.detach())

                loss = torch.nn.MSELoss()
                #disc_cost = torch.mean(disc_fake) - torch.mean(disc_real)
                disc_cost = loss(disc_fake, disc_real)
                disc_cost.backward(retain_graph=True)

                if use_wgan:
                    wgan_loss = compute_gp(discriminator, real_node_inputs.data, real_tau_inputs.data, real_x_inputs.data, \
                                           real_ends, real_t0_res_inputs.data, fake_x_inputs.data, fake_t0_res_inputs.data, \
                                           fake_node_inputs.data, fake_tau_inputs.data, fake_ends.data)
                    #disc_cost = disc_cost + wgan_loss
                    wgan_loss.backward()

                optimizer_D.step()
        print("disc fake & real at epoch {}: {}, {}".format(epoch, torch.mean(disc_fake), torch.mean(disc_real)))
        print("disc_cost: {}".format(disc_cost))
        pbar.close()

        # single run on generator
        for p in generator.parameters():
            p.requires_grad = True
        if use_reward:
            for p in reward.parameters():
                p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = False
        optimizer_G.zero_grad()
        initial_states_noise = make_noise([batch_size, noise_dimension], noise_type)
        initial_states_noise.requires_grad = True

        # generate fake data
        fake_x_inputs, fake_t0_res_inputs, \
        fake_node_inputs, fake_tau_inputs, \
        fake_ends = generator(initial_states_noise)

        if use_reward:
            record_generated_network.append(fake_node_inputs)

        disc_fake = discriminator(fake_x_inputs, fake_t0_res_inputs,
                                  fake_node_inputs, fake_tau_inputs, fake_ends)
        gen_cost = -torch.mean(disc_fake)  # careful here whether a negative sign should be put
        print("gen cost at epoch {}: {}".format(epoch, gen_cost))
        gen_cost.backward()
        optimizer_G.step()

        # single run on reward
        if use_reward:
            if epoch > 0 and epoch % reward_every == 0:
                if epoch <reward_pretraining:
                    for p in generator.parameters():
                        p.requires_grad = False
                else:
                    for p in generator.parameters():
                        p.requires_grad = True
                for p in reward.parameters():
                    p.requires_grad = True
                for p in discriminator.parameters():
                    p.requires_grad = False

                optimizer_R.zero_grad()
                record_generated_network = np.array([r.detach().cpu().numpy() for r in record_generated_network])   # (reward_every,bs,2*rw_len,num_nodes)
                r_e, bs, rw2, _ = record_generated_network.shape
                record_generated_network = record_generated_network.transpose(1, 0, 2, 3).reshape(bs, -1, n_nodes)   # (bs, reward_every*2*rw_len, num_nodes)
                record_generated_network = torch.from_numpy(record_generated_network).cuda()
                reward_score = reward.forward(node_inputs=record_generated_network)   # (bs, 17)
                reward_loss = loss_reward(reward_score.float(), torch.from_numpy(real_score).expand(bs, 17).cuda().float())
                #reward_loss = torch.mean(reward_score.float() - torch.from_numpy(real_score).expand(bs, 17).cuda().float(),1)
                print("reward_score: ", reward_score)
                print("real_score: ", real_score)
                print("reward_loss: ", reward_loss)
                reward_loss.backward()
                optimizer_R.step()

                record_generated_network = []

        if epoch > 0 and epoch % eval_every == 0:
            print("start evaluation for epoch {}".format(epoch))
            generator.eval()
            discriminator.eval()
            n_samples = batch_size * 10
            n_eval_iters = int(eval_transitions / n_samples)  # =n_eval_loop
            fake_graphs = []
            fake_x_t0 = []
            print("n_eval_iters, n_eval_loop: {}, {}".format(n_eval_iters, n_eval_loop))
            pbar = tqdm(total = n_eval_iters)
            for q in range(n_eval_iters):
                pbar.update(1)
                fake_x, fake_t0, fake_edges, fake_t, fake_end = \
                    generate_discrete(generator, n_samples, n_eval_loop)
                node_logit = generator.logit
                smpls = None
                stop = [False] * n_samples
                for i in range(n_eval_loop):
                    x, t0, e, tau, le = fake_x[i], fake_t0[i], fake_edges[i], fake_t[i], fake_end[i]
                    #print("x shape: ", x.shape)     # (bs*10)
                    #print("t0 shape: ", t0.shape)   # (bs*10, 1)
                    #print("e shape: ", e.shape)     # (bs*10, rw_len*2)
                    #print("tau shape: ", tau.shape) # (bs*10, rw_len, 1)
                    #print("le shape: ", le.shape)   # (bs*10)
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
                        #print("smpls0: ", smpls.shape) # (2560, 4, 3)
                    else:
                        new_pred = np.concatenate([e[:, -1:], tau[:, -1:]], axis=-1)
                        #print("new_pred: ", new_pred.shape) # (2560, 1, 3)
                        smpls = np.concatenate([smpls, new_pred], axis=1)
                        #print("smpls1: ", smpls.shape) # (2560, n_eval_loop+3, 3)

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
            pbar.close()

            # reformat fake_graph and remove -1 value
            fake_graphs = np.array(fake_graphs)
            fake_graphs = convert_graphs(fake_graphs)
            fake_graphs[:, 3] = t_end - fake_graphs[:, 3]
            np.save("fake_graphs.npy", fake_graphs)
            Gs = Graphs(test_edges, N=n_nodes, tmax=t_end, edge_contact_time=edge_contact_time)
            FGs = Graphs(fake_graphs, N=n_nodes, tmax=t_end, edge_contact_time=edge_contact_time)
            #try:
            mmd_avg_degree = MMD_Average_Degree_Distribution(Gs, FGs)
            #except:
            #mmd_avg_degree = 1000000
            print('mmd_avg_degree: {}'.format(mmd_avg_degree))
            print(
                'Real Mean_Average_Degree_Distribution: \n{}'.format(Gs.Mean_Average_Degree_Distribution()))
            print(
                'Fake Mean_Average_Degree_Distribution: \n{}'.format(FGs.Mean_Average_Degree_Distribution()))
        
            if mmd_avg_degree < mmd_avg_degree_min:
                mmd_avg_degree_min = min(mmd_avg_degree_min, mmd_avg_degree)
                print("save model to {}".format(save_model_path))
                torch.save(generator.state_dict(), save_model_path + 'generator.pt')
                torch.save(discriminator.state_dict(), save_model_path + 'discriminator.pt')
                torch.save(reward.state_dict(), save_model_path + 'reward.pt')
        if abs(gen_cost) < early_stopping and abs(disc_cost) < early_stopping:
            print("Stop the training process because the gen cost and disc cost is smaller than the creteria...")
            break


if __name__ == "__main__":
    # read config
    config = configparser.ConfigParser()
    config.readfp(open(configFilePath))

    run(config, args)
