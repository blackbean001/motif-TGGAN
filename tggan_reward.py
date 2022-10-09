# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:01:53 2022

@author: Administrator
"""

import os
import shutil
import logging
import datetime
import time
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

"""
# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
if not os.path.isdir('logs'):
    os.makedirs('logs')
log_file = 'logs/log_{}.log'.format(datetime.datetime.now().strftime("%Y_%B_%d_%I-%M-%S%p"))
open(log_file, 'a').close()
# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
"""


logging.info('is GPU available? {}'.format(torch.cuda.is_available()))
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Generator(nn.Module):
    def __init__(self, N, rw_len,
                 t_end, device,
                 batch_size=128, noise_dim=16,
                 noise_type="Gaussian",
                 use_gumbel=True,
                 use_decoder='normal',
                 generator_x_up_layers=[64],
                 generator_t0_up_layers=[128],
                 generator_tau_up_layers=[128],
                 constraint_method='min_max',
                 generator_layers=[40],
                 W_down_generator_size=128):
        super(Generator, self).__init__()
        self.params = {
            'noise_dim': noise_dim,
            'noise_type': noise_type,
            't_end': t_end,
            'generator_x_up_layers': generator_x_up_layers,
            'generator_t0_up_layers': generator_t0_up_layers,
            'generator_tau_up_layers': generator_tau_up_layers,
            'constraint_method': constraint_method,
            'Generator_Layers': generator_layers,
            'W_Down_Generator_size': W_down_generator_size,
            'batch_size': batch_size,
            'use_gumbel': use_gumbel,
            'use_decoder': use_decoder}

        self.N = N
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.t_end = t_end
        self.device = device
        self.use_gumble = use_gumbel
        self.noise_dim = self.params['noise_dim']
        self.G_layers = self.params['Generator_Layers']
        self.temp = 1
        self.G_x_up_layers = self.params['generator_x_up_layers']
        self.G_t0_up_layers = self.params['generator_t0_up_layers']
        self.G_tau_up_layers = self.params['generator_tau_up_layers']

        # stacked_lstm
        self.noise_preprocessing_module = []
        self.noise_preprocessing_module_h = []
        self.noise_preprocessing_module_c = []
        self.stacked_lstm_module = []
        for ix, size in enumerate(self.G_layers):
            self.noise_preprocessing_module += [nn.Linear(self.noise_dim, size), nn.Tanh()]
            self.noise_preprocessing_module_h += [nn.Linear(size, size), nn.Tanh()]
            self.noise_preprocessing_module_c += [nn.Linear(size, size), nn.Tanh()]
            if ix == 0:
                self.stacked_lstm_module.append(nn.LSTM(self.params['W_Down_Generator_size'], size))
            else:
                self.stacked_lstm_module.append(self.G_layers[ix - 1], size)
        self.stacked_lstm_module = nn.ModuleList(self.stacked_lstm_module)
        self.noise_preprocessing_module = nn.ModuleList(self.noise_preprocessing_module)
        self.noise_preprocessing_module_h = nn.ModuleList(self.noise_preprocessing_module_h)
        self.noise_preprocessing_module_c = nn.ModuleList(self.noise_preprocessing_module_c)

        # G_x_up_layers
        self.G_x_up_module1 = []
        for ix, size in enumerate(self.G_x_up_layers):
            if ix == 0:
                self.G_x_up_module1 += [nn.Linear(self.G_layers[-1], size), nn.Tanh()]
            else:
                self.G_x_up_modul1 += [nn.Linear(self.G_x_up_layers[ix - 1], size), nn.Tanh()]
        self.G_x_up_module1 += [nn.Linear(self.G_x_up_layers[-1], 2)]
        self.G_x_up_module1 = nn.ModuleList(self.G_x_up_module1)
        self.G_x_up_module2 = nn.ModuleList([nn.Linear(2, 1), \
                                             nn.Linear(1, self.params['W_Down_Generator_size']), \
                                             nn.Tanh()])
        self.G_x_up_softmax = nn.Softmax()

        self.G_t0_up_module_loc = []
        self.G_t0_up_module_scale = []
        for ix, size in enumerate(self.G_t0_up_layers):
            if ix == 0:
                self.G_t0_up_module_loc += [nn.Linear(self.G_layers[-1], size), nn.Tanh()]
                self.G_t0_up_module_scale += [nn.Linear(self.G_layers[-1], size), nn.Tanh()]
            else:
                self.G_t0_up_module_loc += [nn.Linear(size, size), nn.Tanh()]
                self.G_t0_up_module_scale += [nn.Linear(size, size), nn.Tanh()]
        self.G_t0_up_module_loc.append(nn.Linear(size, 1))
        self.G_t0_up_module_scale.append(nn.Linear(size, 1))
        self.G_t0_up_module_loc = nn.ModuleList(self.G_t0_up_module_loc)
        self.G_t0_up_module_scale = nn.ModuleList(self.G_t0_up_module_scale)

        self.t0_res2input_linear = [nn.Linear(1, self.params['W_Down_Generator_size']), nn.Tanh()]
        self.t0_res2input_linear = nn.ModuleList(self.t0_res2input_linear)

        self.edge_W_up1 = nn.Linear(self.G_layers[-1], self.N, bias=True)
        self.edge_up_softmax = nn.Softmax()
        self.edge_W_up2 = nn.Linear(self.N, self.params['W_Down_Generator_size'], bias=True)
        self.edge_W_up3 = nn.ModuleList([nn.Linear(1, self.params['W_Down_Generator_size'], bias=True), nn.Tanh()])

        # G_y_up_layers
        self.G_y_up_module1 = []
        for ix, size in enumerate(self.G_x_up_layers):
            if ix == 0:
                self.G_y_up_module1 += [nn.Linear(self.G_layers[-1], size), nn.Tanh()]
            else:
                self.G_y_up_module1 += [nn.Linear(self.G_x_up_layers[ix - 1], size), nn.Tanh()]
        self.G_y_up_module1 += [nn.Linear(self.G_x_up_layers[-1], 2)]
        self.G_y_up_module1 = nn.ModuleList(self.G_y_up_module1)
        self.G_y_up_softmax = nn.Softmax()

    def stacked_lstm(self, inputs, z, state=None):
        if len(inputs.size()) != 3:
            inputs = torch.unsqueeze(inputs, 0)
        #print("lstm inputs size: ", inputs.size())
        for i, layer in enumerate(self.stacked_lstm_module):
            #print("lstm layer: ", layer)
            if state == None:
                if z is None:
                    initial_states_noise = make_noise([n_samples, self.noise_dim], self.params['noise_type'])
                else:
                    initial_states_noise = z
                initial_states_noise = initial_states_noise.to(self.device)
                initial_states_noise = Variable(initial_states_noise, volatile=True)
                intermediate = self.noise_preprocessing_module[2 * i](initial_states_noise)
                intermediate = self.noise_preprocessing_module[2 * i + 1](intermediate)
                h = self.noise_preprocessing_module_h[2 * i](intermediate)
                h = self.noise_preprocessing_module_h[2 * i + 1](h)
                c = self.noise_preprocessing_module_c[2 * i](intermediate)
                c = self.noise_preprocessing_module_c[2 * i + 1](c)
                if len(h.size()) != 3:
                    h = torch.unsqueeze(h, 0)
                    c = torch.unsqueeze(c, 0)
            else:
                h, c = state
                if len(h.size()) != 3:
                    h = torch.unsqueeze(h, 0)
                    c = torch.unsqueeze(c, 0)
            inputs, (h, c) = self.stacked_lstm_module[i](inputs, (h, c))
        output, state = inputs, (h, c)
        return output, state

    def generate_x(self, x_logit):
        #print("x_logit: ", x_logit.size())
        #print("self.G_x_up_module1: ", self.G_x_up_module1)
        for i, size in enumerate(self.G_x_up_layers):
            x_logit = self.G_x_up_module1[2 * i](x_logit)
            x_logit = self.G_x_up_module1[2 * i + 1](x_logit)
        x_logit = self.G_x_up_module1[-1](x_logit)
        if self.use_gumble:
            x_output = F.gumbel_softmax(x_logit, tau=self.temp, hard=True)
        else:
             x_output = self.G_x_up_softmax(x_logit)
        return x_output

    def generate_y(self, end_logit):
        for i, size in enumerate(self.G_x_up_layers):
            end_logit = self.G_y_up_module1[2 * i](end_logit)
            end_logit = self.G_y_up_module1[2 * i + 1](end_logit)
        end_logit = self.G_y_up_module1[-1](end_logit)
        if self.use_gumble:
            y_output = F.gumbel_softmax(end_logit, tau=self.temp, hard=True)
        else:
            y_output = self.G_y_up_softmax(end_logit)
        return y_output

    def generate_time(self, output):
        n_samples = list(output.size())[0]
        if self.params['use_decoder'] == 'normal':
            #print("t_output: ", output.size(), output)
            loc_t0 = output
            scale_t0 = output
            for ix, layer in enumerate(self.G_t0_up_layers):
                loc_t0 = self.G_t0_up_module_loc[2 * ix](loc_t0)
                loc_t0 = self.G_t0_up_module_loc[2 * ix + 1](loc_t0)
                scale_t0 = self.G_t0_up_module_scale[2 * ix](scale_t0)
                scale_t0 = self.G_t0_up_module_scale[2 * ix + 1](scale_t0)

            loc_t0 = self.G_t0_up_module_loc[-1](loc_t0)
            scale_t0 = self.G_t0_up_module_scale[-1](scale_t0)
            if len(loc_t0.size()) == 3:
                loc_t0 = loc_t0[-1,:,:]
                scale_t0 = scale_t0[-1,:,:]
            #print("loc_t0: ", loc_t0)
            #print("scale_t0: ", scale_t0)
            t0_wait = []
            for i in range(self.sample_n):
                t0_wait.append(truncated_normal_(torch.tensor([1.]).to(self.device), mean=loc_t0[i, 0],
                                         std=scale_t0[i, 0]))
            t0_wait = torch.stack(t0_wait, 0)

        return t0_wait

    def forward(self, z, x_input=None, t0_input=None,
                edge_input=None, tau_input=None, gumbel=True):
        #print("z: ", z)
        #print("x_input: ", x_input)
        self.sample_n = z.size()[0]
        # noise preprocessing
        inputs = torch.zeros(self.sample_n, self.params['W_Down_Generator_size']).to(self.device)  # (batch, emb_size); emb_size =  num_nodes // 2
        output, state = self.stacked_lstm(inputs, z)
        # output: (L, bs, size)
        # generate start x
        if x_input == None:
            x_output = self.generate_x(output[-1,:,:])
        else:
            x_output = x_input.float()   # (bs, 2)
        # convert to inputs
        x_down = self.G_x_up_module2[0](x_output)
        #print("x_down: ", x_down)
        inputs = self.G_x_up_module2[1](x_down)
        #print("i: ", inputs)
        inputs = self.G_x_up_module2[2](inputs)  # (bs, emb_size)
        #print("inputs0: ", inputs)
        # generate nodes and time
        node_outputs, tau_outputs = [], []
        # generate the first three start elements: start x, residual time,
        # and maximum possible length
        output, state = self.stacked_lstm(inputs, z, state=state)
        #print("output: ", output.size(), output)
        #print("state: ", state)
        #print("t0_input: ", t0_input)
        # generate start time
        if t0_input != None:
            t0_res_output = t0_input
        else:
            t0_res_output = self.generate_time(output[-1,:,:])
        #print("t0_res_output0: ", t0_res_output)
        if self.params['constraint_method'] != "none":
             t0_res_output = time_constraint(t0_res_output, \
                                                  method=self.params['constraint_method']) * self.t_end
        #print("t0_res_output1: ", t0_res_output)
        #print('t0_res_output1: ', t0_res_output.size())
        condition = torch.eq(torch.argmax(x_output, -1), 1)
        #print("condition: ", condition.size())
        condition = torch.unsqueeze(condition, 1)
        t0_res_output = torch.where(condition, \
                                    torch.ones_like(t0_res_output), t0_res_output)
        res_time = t0_res_output
        #print('t0_res_output2: ', t0_res_output.size())
        # convert to input
        inputs = self.t0_res2input_linear[0](t0_res_output)
        inputs = torch.unsqueeze(inputs, 0)
        #print("inputs: ", inputs.size())
        # generate temporal edge part
        for i in range(self.rw_len):
            for j in range(2):
                # lstm for first edge
                output, state = self.stacked_lstm(inputs, z,  state=state)
                if edge_input != None and i <= self.rw_len - 2:
                    output = torch.unsqueeze(edge_input[:, i * 2 + j, :], 0)
                else:
                    logit = self.edge_W_up1(output)
                    self.logit = logit
                    if self.use_gumble:
                        output = F.gumbel_softmax(logit, tau=self.temp, hard=True)
                    else:
                        output = self.edge_up_softmax(logit)
                node_outputs.append(output)
                inputs = self.edge_W_up2(output)
            # lstm for tau
            output, state = self.stacked_lstm(inputs, z, state=state)
            if tau_input != None and i <= self.rw_len - 2:
                tau = tau_input[:, i]
            else:
                tau = self.generate_time(output[-1,:,:])
                if self.params['constraint_method'] != "none":
                    tau = time_constraint(tau, method=self.params['constraint_method']) * res_time
            tau_outputs.append(tau)
            inputs = self.edge_W_up3[0](tau)
            inputs = self.edge_W_up3[1](inputs)

        # lstm for end indicator
        output, state = self.stacked_lstm(inputs, z, state=state)
        end_outputs = self.generate_y(output)

        node_outputs = torch.stack(node_outputs, 1)
        tau_outputs = torch.stack(tau_outputs, 1)
        #print("node_outputs: ", node_outputs.size())
        node_outputs = node_outputs[0,:,:,:].permute(1,0,2) # (bs,2*rw_len,num_nodes)
        end_outputs = end_outputs[0,:,:]

        return x_output, t0_res_output, node_outputs, tau_outputs, end_outputs


class Discriminator(nn.Module):
    def __init__(self, N, rw_len,
                 discriminator_layers=[30],
                 W_down_discriminator_size=128):
        super(Discriminator, self).__init__()

        self.params = {
                'D_layers': discriminator_layers,
                'W_Down_Discriminator_size': W_down_discriminator_size
        }

        self.N = N
        self.rw_len = rw_len
        self.D_layers = discriminator_layers
        self.disc_x_module = nn.ModuleList([nn.Linear(2, 1),
                                            nn.Linear(1, self.params['W_Down_Discriminator_size']),
                                            nn.Tanh()])

        self.disc_t0_module = nn.ModuleList([nn.Linear(1, self.params['W_Down_Discriminator_size']), \
                                             nn.Tanh()])

        self.disc_nodes_module = nn.Linear(self.N, self.params['W_Down_Discriminator_size'])
        self.disc_tau_module = nn.Linear(1, self.params['W_Down_Discriminator_size'])

        self.disc_end_module = nn.ModuleList([nn.Linear(2, 1),
                                              nn.Linear(1, self.params['W_Down_Discriminator_size']), \
                                              nn.Tanh()])

        self.stacked_lstm_module = []
        for ix, size in enumerate(self.D_layers):
            if ix == 0:
                self.stacked_lstm_module.append(nn.LSTM(self.params['W_Down_Discriminator_size'], size))
            else:
                self.stacked_lstm_module.append(nn.LSTM(self.D_layers[ix - 1], size))
        self.stacked_lstm_module = nn.ModuleList(self.stacked_lstm_module)

        self.final_score_dense = nn.Linear(self.D_layers[-1], 1)

    def forward(self, x, t0_res, node_inputs, tau_inputs, end):
        '''
        x: (bs, 2)
        t0_res: (bs, 1)
        node_inputs: (bs, rw_len*2)
        tau_inputs: (bs, 1)
        end: (bs, 2)
        '''
        #x = Variable(x, requires_grad=True)
        #t0_res = Variable(t0_res, requires_grad=True)
        #node_inputs = Variable(node_inputs, requires_grad=True)
        #tau_inputs = Variable(tau_inputs, requires_grad=True)
        #end = Variable(end, requires_grad=True)


        # discrimator for x
        x_input_reshape = torch.reshape(x, (-1, 2)).type(torch.float)
        for layer in self.disc_x_module:
            x_input_reshape = layer(x_input_reshape)  # tensor(bs, 128)

        # discriminator for t0
        t0_inputs = torch.reshape(t0_res, (-1, 1)).type(torch.float)
        for layer in self.disc_t0_module:
            t0_inputs = layer(t0_inputs)
        t0_input_up = t0_inputs  # tensor(bs, 128)

        # discriminator for nodes
        node_input_reshape = torch.reshape(node_inputs, (-1, self.N)).type(torch.float)
        node_output = self.disc_nodes_module(node_input_reshape)
        node_output = torch.reshape(node_output, (-1, 2 * self.rw_len, self.params['W_Down_Discriminator_size']))
        node_output = torch.unbind(node_output, 1)  # [tensor(bs,128)]*(2*self.rw_len)

        # discriminator for tau
        tau_input_reshape = torch.reshape(tau_inputs, (-1, 1)).type(torch.float) # (bs*rw_len, 1)
        tau_output = self.disc_tau_module(tau_input_reshape) # (bs*rw_len, emb_size)
        tau_output = torch.reshape(tau_output, (-1, self.rw_len, self.params['W_Down_Discriminator_size']))    # (bs, 4, emb_size)
        tau_output = torch.unbind(tau_output, 1)  # [tensor(bs,128)]*(self.rw_len)

        # discriminator for y
        end_input_reshape = torch.reshape(end, (-1, 2)).type(torch.float)
        for layer in self.disc_end_module:
            end_input_reshape = layer(end_input_reshape)  # tensor(bs,128)

        # generate inputs for lstm
        x_input_reshape = torch.unsqueeze(x_input_reshape, 0)
        t0_input_up = torch.unsqueeze(t0_input_up, 0)
        inputs = torch.cat((x_input_reshape, t0_input_up), 0)
        for i in range(self.rw_len):
            A = torch.unsqueeze(node_output[i * 2], 0)  # (1, bs, emb_size)
            B = torch.unsqueeze(node_output[i * 2 + 1], 0)  # (1, bs, emb_size)
            C = torch.unsqueeze(tau_output[i], 0)  # (1, bs, emb_size)
            inputs = torch.cat((inputs, A, B, C), 0)
        end_input_reshape = torch.unsqueeze(end_input_reshape, 0)
        inputs = torch.cat((inputs, end_input_reshape), 0)  # inputs: (3*rw_len+3, bs, emb_size)
        for ix, layer in enumerate(self.stacked_lstm_module):
            if ix == 0:
                inputs, (hn, cn) = layer(inputs)
            else:
                inputs, (hn, cn) = layer(inputs, (hn, cn))
        outputs = inputs
        last_output = outputs[-1] # (bs, self.D_layers[-1])
        final_score = self.final_score_dense(last_output)

        return final_score


class RewardNet(nn.Module):
    def __init__(self, N, rw_len, reward_every,
                 reward_layers=[15],
                 W_down_reward_size=64,
                 num_motif=17):
        super(RewardNet, self).__init__()

        self.params = {
                'R_layers': reward_layers,
                'W_Down_Reward_size': W_down_reward_size
        }

        self.N = N
        self.rw_len = rw_len
        self.reward_every = reward_every
        self.num_motif = num_motif
        self.R_layers = reward_layers
        self.reward_x_module = nn.ModuleList([nn.Linear(2, 1),
                                            nn.Linear(1, self.params['W_Down_Reward_size']),
                                            nn.Tanh()])

        self.reward_t0_module = nn.ModuleList([nn.Linear(1, self.params['W_Down_Reward_size']), \
                                             nn.Tanh()])

        self.reward_nodes_module = nn.Linear(self.N, self.params['W_Down_Reward_size'])
        self.reward_tau_module = nn.Linear(1, self.params['W_Down_Reward_size'])

        self.reward_end_module = nn.ModuleList([nn.Linear(2, 1),
                                              nn.Linear(1, self.params['W_Down_Reward_size']), \
                                              nn.Tanh()])

        self.stacked_lstm_module = []
        for ix, size in enumerate(self.R_layers):
            if ix == 0:
                self.stacked_lstm_module.append(nn.LSTM(self.params['W_Down_Reward_size'], size))
            else:
                self.stacked_lstm_module.append(nn.LSTM(self.R_layers[ix - 1], size))
        self.stacked_lstm_module = nn.ModuleList(self.stacked_lstm_module)

        self.final_score_dense = nn.Linear(self.R_layers[-1], self.num_motif)

    def forward(self, x=None, t0_res=None, node_inputs=None, tau_inputs=None, end=None):
        '''
        x: (bs, 2)
        t0_res: (bs, 1)
        node_inputs: (bs, reward_every*rw_len*2, n_nodes)
        tau_inputs: (bs, 1)
        end: (bs, 2)
        '''

        if x != None:
            # reward for x
            x_input_reshape = torch.reshape(x, (-1, 2)).type(torch.float)
            for layer in self.reward_x_module:
                x_input_reshape = layer(x_input_reshape)  # tensor(bs, 128)

        if t0_res != None:
            # reward for t0
            t0_inputs = torch.reshape(t0_res, (-1, 1)).type(torch.float)
            for layer in self.reward_t0_module:
                t0_inputs = layer(t0_inputs)
            t0_input_up = t0_inputs  # tensor(bs, 128)

        # reward for nodes
        node_input_reshape = torch.reshape(node_inputs, (-1, self.N)).type(torch.float)
        node_output = self.reward_nodes_module(node_input_reshape)
        node_output = torch.reshape(node_output, (-1, 2 * self.rw_len * self.reward_every, self.params['W_Down_Reward_size']))
        node_output = torch.unbind(node_output, 1)  # [tensor(bs,128)]*(2*self.rw_len*self.reward_every)

        if tau_inputs != None:
            # reward for tau
            tau_input_reshape = torch.reshape(tau_inputs, (-1, 1)).type(torch.float) # (bs*rw_len*reward_every, 1)
            tau_output = self.reward_tau_module(tau_input_reshape) # (bs*rw_len*reward_every, emb_size)
            tau_output = torch.reshape(tau_output, (-1, self.rw_len, self.params['W_Down_Reward_size']))    # (bs, 4, emb_size)
            tau_output = torch.unbind(tau_output, 1)  # [tensor(bs,128)]*(self.rw_len*reward_every)

        if end != None:
            # reward for y
            end_input_reshape = torch.reshape(end, (-1, 2)).type(torch.float)
            for layer in self.reward_end_module:
                end_input_reshape = layer(end_input_reshape)  # tensor(bs,128)

        if x != None:
            # generate inputs for lstm
            x_input_reshape = torch.unsqueeze(x_input_reshape, 0)
            t0_input_up = torch.unsqueeze(t0_input_up, 0)
            inputs = torch.cat((x_input_reshape, t0_input_up), 0)
            for i in range(self.rw_len * self.reward_every):
                A = torch.unsqueeze(node_output[i * 2], 0)  # (1, bs, emb_size)
                B = torch.unsqueeze(node_output[i * 2 + 1], 0)  # (1, bs, emb_size)
                C = torch.unsqueeze(tau_output[i], 0)  # (1, bs, emb_size)
                inputs = torch.cat((inputs, A, B, C), 0)
            end_input_reshape = torch.unsqueeze(end_input_reshape, 0)
            inputs = torch.cat((inputs, end_input_reshape), 0)  # inputs: (3*rw_len+3, bs, emb_size)
        else:
            for i in range(self.rw_len * self.reward_every):
                if i==0:
                    A = torch.unsqueeze(node_output[i * 2], 0)  # (1, bs, emb_size)
                    B = torch.unsqueeze(node_output[i * 2 + 1], 0)  # (1, bs, emb_size)
                    inputs = torch.cat((A, B), 0)
                else:
                    A = torch.unsqueeze(node_output[i * 2], 0)  # (1, bs, emb_size)
                    B = torch.unsqueeze(node_output[i * 2 + 1], 0)  # (1, bs, emb_size)
                    inputs = torch.cat((inputs, A, B), 0)

        # run self.stacked_lstm_module layer
        for ix, layer in enumerate(self.stacked_lstm_module):
            if ix == 0:
                inputs, (hn, cn) = layer(inputs)
            else:
                inputs, (hn, cn) = layer(inputs, (hn, cn))

        outputs = inputs
        last_output = outputs[-1] # (bs, self.D_layers[-1])
        final_score = self.final_score_dense(last_output)   # (bs, 17)

        return final_score