"""
2021: aleatoryscience
Lys Sanz Moreta
Draupnir : GP prior VAE for Ancestral Sequence Resurrection
=======================
"""
from abc import ABC, abstractmethod
# TORCH
import torch.nn as nn
import math
from pyro.infer import SVI
from ignite.engine import Engine, Events
from abc import ABC, abstractmethod
from typing import Callable
import torch
from pyro import distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints


class RNNEncoder(nn.Module):
    def __init__(self, max_seq_len,aa_prob,n_leaves,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNEncoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.linear_means = nn.Linear(self.gru_hidden_dim, self.z_dim)
        self.linear_std = nn.Linear(self.gru_hidden_dim, self.z_dim)

        self.softplus = nn.Softplus()

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)


    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,max_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        rnn_output = self.fc1(rnn_output[:,-1]) #pick the last state of the sequence given by the GRU
        output_means = self.linear_means(rnn_output)
        output_std = self.softplus(self.linear_std(rnn_output))
        return output_means,output_std


class RNNEncoder_no_mean(nn.Module):
    def __init__(self, max_seq_len,aa_prob,n_leaves,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNEncoder_no_mean, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        #self.linear_means = nn.Linear(self.gru_hidden_dim, self.z_dim)
        self.linear_std = nn.Linear(self.gru_hidden_dim, self.z_dim)

        self.softplus = nn.Softplus()

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)


    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,max_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        rnn_output = self.fc1(rnn_output[:,-1]) #pick the last state of the sequence given by the GRU
        #output_means = self.linear_means(rnn_output)
        output_std = self.softplus(self.linear_std(rnn_output))
        return output_std




class MLP(nn.Module):
    def __init__(self, max_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(MLP, self).__init__()
        self.hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if pretrained_params is not None:
            self.fc1 = nn.Linear(self.rnn_input_size, self.hidden_dim)
            self.fc1.weight = nn.Parameter(pretrained_params["decoder.fc1.weight"],requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["decoder.fc1.bias"], requires_grad=False)
            self.fc2 = nn.Linear(self.hidden_dim, 2*self.hidden_dim)
            self.fc2.weight = nn.Parameter(pretrained_params["decoder.fc2.weight"],requires_grad=False)
            self.fc2.bias = nn.Parameter(pretrained_params["decoder.fc2.bias"], requires_grad=False)
            self.fc3 = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
            self.fc3.weight = nn.Parameter(pretrained_params["decoder.fc3.weight"],requires_grad=False)
            self.fc3.bias = nn.Parameter(pretrained_params["decoder.fc3.bias"], requires_grad=False)
            self.fc4 = nn.Linear(2*self.hidden_dim, self.hidden_dim)
            self.fc4.weight = nn.Parameter(pretrained_params["decoder.fc4.weight"],requires_grad=False)
            self.fc4.bias = nn.Parameter(pretrained_params["decoder.fc4.bias"], requires_grad=False)
            #self.fc1.train(False)
            self.linear_probs = nn.Linear(self.hidden_dim, self.aa_prob)
            self.linear_probs.weight = nn.Parameter(pretrained_params["decoder.linear_probs.weight"],requires_grad=False)
            self.linear_probs.bias = nn.Parameter(pretrained_params["decoder.linear_probs.bias"],requires_grad=False)
            #self.linear_probs.train(False)

        else:
            self.fc1 = nn.Linear(self.rnn_input_size, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, 2*self.hidden_dim)
            self.fc3 = nn.Linear(2 * self.hidden_dim, 2*self.hidden_dim)
            self.fc4 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.linear_probs = nn.Linear(self.hidden_dim, self.aa_prob)

    def forward(self, input):

        MLP_out = self.fc4(self.fc3(self.fc2(self.fc1(input))))
        output_logits = self.logsoftmax(self.linear_probs(MLP_out))  # [n_nodes,max_seq_len,aa_probs]
        return output_logits


class RNNDecoder_Tiling(nn.Module):
    def __init__(self, max_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if pretrained_params is not None:
            self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1.weight = nn.Parameter(pretrained_params["decoder.fc1.weight"],requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["decoder.fc1.bias"], requires_grad=False)
            #self.fc1.train(False)
            self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.linear_probs.weight = nn.Parameter(pretrained_params["decoder.linear_probs.weight"],requires_grad=False)
            self.linear_probs.bias = nn.Parameter(pretrained_params["decoder.linear_probs.bias"],requires_grad=False)
            #self.linear_probs.train(False)
            self.rnn = nn.GRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)
            #self.rnn.__setstate__(pretrained_params) #Is this working?
            self.rnn.weight_ih_l0 = nn.Parameter(pretrained_params["decoder.rnn.weight_ih_l0"],requires_grad=False)
            self.rnn.weight_hh_l0 = nn.Parameter(pretrained_params["decoder.rnn.weight_hh_l0"],requires_grad=False)
            self.rnn.bias_ih_l0 = nn.Parameter(pretrained_params["decoder.rnn.bias_ih_l0"],requires_grad=False)
            self.rnn.bias_hh_l0 = nn.Parameter(pretrained_params["decoder.rnn.bias_hh_l0"],requires_grad=False)
            self.rnn.weight_ih_l0_reverse =nn.Parameter(pretrained_params["decoder.rnn.weight_ih_l0_reverse"],requires_grad=False)
            self.rnn.weight_hh_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.weight_hh_l0_reverse"],requires_grad=False)
            self.rnn.bias_ih_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.bias_ih_l0_reverse"],requires_grad=False)
            self.rnn.bias_hh_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.bias_hh_l0_reverse"],requires_grad=False)

        else:
            self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.rnn = nn.GRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)
            #rnn.state_dict()


    def forward(self, input, hidden):
        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,max_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output_logits = self.logsoftmax(self.linear_probs(self.fc1(rnn_output)))  # [n_nodes,max_seq_len,aa_probs]
        return output_logits


class TransformerDecoder_Tiling(nn.Module):
    def __init__(self, max_seq_len, aa_prob, gru_hidden_dim, z_dim, rnn_input_size, kappa_addition, num_layers,
                 pretrained_params):
        super(TransformerDecoder_Tiling, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if pretrained_params is not None:
            self.fc1 = nn.Linear(self.rnn_input_size ,self.gru_hidden_dim)
            self.fc1.weight = nn.Parameter(pretrained_params["decoder.fc1.weight"], requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["decoder.fc1.bias"], requires_grad=False)
            self.linear_probs = nn.Linear(self.rnn_input_size, self.aa_prob)
            self.linear_probs.weight = nn.Parameter(pretrained_params["decoder.linear_probs.weight"],requires_grad=False)
            self.linear_probs.bias = nn.Parameter(pretrained_params["decoder.linear_probs.bias"], requires_grad=False)
            #TRANSFORMER#
            self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.rnn_input_size, nhead=1,dim_feedforward=2048, dropout=0.1, activation="relu")
            self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)

            self.transformer_decoder.linear1.weight = nn.Parameter(pretrained_params["decoder.transformer_decoder.linear1.weight"], requires_grad=False)
            self.transformer_decoder.linear1.bias = nn.Parameter(pretrained_params["decoder.transformer_decoder.linear1.bias"], requires_grad=False)
            self.transformer_decoder.linear2.weight = nn.Parameter(pretrained_params["decoder.transformer_decoder.linear2.weight"], requires_grad=False)
            self.transformer_decoder.linear2.bias = nn.Parameter(pretrained_params["decoder.transformer_decoder.linear2.bias"], requires_grad=False)
            self.transformer_decoder.norm1.weight = nn.Parameter(pretrained_params["decoder.transformer_decoder.norm1.weight"],requires_grad=False)
            self.transformer_decoder.norm1.bias = nn.Parameter(pretrained_params["decoder.transformer_decoder.norm1.bias"], requires_grad=False)
            self.transformer_decoder.norm2.weight = nn.Parameter(pretrained_params["decoder.transformer_decoder.norm2.weight"],requires_grad=False)
            self.transformer_decoder.norm2.bias = nn.Parameter(pretrained_params["decoder.transformer_decoder.norm2.bias"], requires_grad=False)
            self.transformer_decoder.norm3.weight = nn.Parameter(pretrained_params["decoder.transformer_decoder.norm3.weight"],requires_grad=False)
            self.transformer_decoder.norm3.bias = nn.Parameter(pretrained_params["decoder.transformer_decoder.norm3.bias"], requires_grad=False)



        else:
            self.fc1 = nn.Linear(self.rnn_input_size, self.rnn_input_size)
            self.linear_probs = nn.Linear(self.rnn_input_size, self.aa_prob)
            self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.max_seq_len, nhead=1,dim_feedforward=self.gru_hidden_dim, dropout=0.1, activation="relu") #d_model / n_heads
            self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer,num_layers=self.num_layers,)

    def forward(self, sequences, latent):
        #sequences.unsqueeze(-1).permute(1,0,2),
        transformer_output = self.transformer_decoder(tgt=latent.permute(2,0,1),tgt_mask=None, memory=latent.permute(2,0,1))  #TODO: Are the targets the sequences and memory the latent_space?
        output_logits = self.logsoftmax(self.linear_probs(self.fc1(transformer_output.permute(1,2,0))))  # [n_nodes,max_seq_len,aa_probs]
        return output_logits


class SRUDecoder_Tiling(nn.Module):
    def __init__(self, max_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(SRUDecoder_Tiling, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if pretrained_params is not None:
            self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1.weight = nn.Parameter(pretrained_params["decoder.fc1.weight"],requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["decoder.fc1.bias"], requires_grad=False)

            # self.fc2 = nn.Linear(self.gru_hidden_dim, self.gru_hidden_dim)
            # self.fc2.weight = nn.Parameter(pretrained_params["decoder.fc2.weight"],requires_grad=False)
            # self.fc2.bias = nn.Parameter(pretrained_params["decoder.fc2.bias"], requires_grad=False)

            self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.linear_probs.weight = nn.Parameter(pretrained_params["decoder.linear_probs.weight"],requires_grad=False)
            self.linear_probs.bias = nn.Parameter(pretrained_params["decoder.linear_probs.bias"],requires_grad=False)

            self.sru = SRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)
            self.sru.rnn_lst.weight= nn.Parameter(pretrained_params["decoder$$$sru.rnn_lst.0.weight"],requires_grad=False)
            self.sru.rnn_lst.weight_c = nn.Parameter(pretrained_params["decoder$$$sru.rnn_lst.0.weight_c"],requires_grad=False)
            self.sru.rnn_lst.bias =  nn.Parameter(pretrained_params["ecoder$$$sru.rnn_lst.0.bias"],requires_grad=False)

        else:
            self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            #self.fc2 = nn.Linear(self.gru_hidden_dim,self.gru_hidden_dim)
            self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.sru = SRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)


    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.sru(input.permute(1,0,2))  # [n_nodes,max_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        output_logits = self.logsoftmax(self.linear_probs(self.fc1(rnn_output.permute(1,0,2))))  # [n_nodes,max_seq_len,aa_probs]
        return output_logits
#

class RNNDecoder_Tiling_Angles(nn.Module):
    def __init__(self, max_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling_Angles, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = 4
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        if pretrained_params is not None:
            #common linear layer
            self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1.weight = nn.Parameter(pretrained_params["decoder.fc1.weight"],requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["decoder.fc1.bias"], requires_grad=False)
            #linear layer logits/probs
            self.fc2_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.fc2_probs.weight = nn.Parameter(pretrained_params["decoder.fc2_probs.weight"],requires_grad=False)
            self.fc2_probs.bias = nn.Parameter(pretrained_params["decoder.fc2_probs.bias"],requires_grad=False)
            #linear layer angles means
            self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_means.weight = nn.Parameter(pretrained_params["decoder.fc2_means.weight"],requires_grad=False)
            self.fc2_means.bias = nn.Parameter(pretrained_params["decoder.fc2_means.bias"], requires_grad=False)
            #linear layer angles kappas
            self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_kappas.weight = nn.Parameter(pretrained_params["decoder.fc2_kappas.weight"],requires_grad=False)
            self.fc2_kappas.bias = nn.Parameter(pretrained_params["decoder.fc2_kappas.bias"], requires_grad=False)
            #RNN
            self.rnn = nn.GRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)
            #self.rnn.__setstate__(pretrained_params) #Is this working?
            self.rnn.weight_ih_l0 = nn.Parameter(pretrained_params["decoder.rnn.weight_ih_l0"],requires_grad=False)
            self.rnn.weight_hh_l0 = nn.Parameter(pretrained_params["decoder.rnn.weight_hh_l0"],requires_grad=False)
            self.rnn.bias_ih_l0 = nn.Parameter(pretrained_params["decoder.rnn.bias_ih_l0"],requires_grad=False)
            self.rnn.bias_hh_l0 = nn.Parameter(pretrained_params["decoder.rnn.bias_hh_l0"],requires_grad=False)
            self.rnn.weight_ih_l0_reverse =nn.Parameter(pretrained_params["decoder.rnn.weight_ih_l0_reverse"],requires_grad=False)
            self.rnn.weight_hh_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.weight_hh_l0_reverse"],requires_grad=False)
            self.rnn.bias_ih_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.bias_ih_l0_reverse"],requires_grad=False)
            self.rnn.bias_hh_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.bias_hh_l0_reverse"],requires_grad=False)
            #self.rnn.train(False)

        else:
            self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc2_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
            self.rnn = nn.GRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)

    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,max_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output = self.fc1(rnn_output)
        output_logits = self.logsoftmax(self.fc2_probs((output)))  # [n_nodes,max_seq_len,aa_probs]
        output_means = self.tanh(self.fc2_means(output))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(output))
        return output_logits,output_means,output_kappas
class RNNDecoder_Angles_Single_SRU(nn.Module):
    def __init__(self, max_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Angles_Single_SRU, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = 6
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        if pretrained_params is not None:
            #linear layer logits
            self.fc1_logits = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1_logits.weight = nn.Parameter(pretrained_params["decoder.fc1_logits.weight"],requires_grad=False)
            self.fc1_logits.bias = nn.Parameter(pretrained_params["decoder.fc1_logits.bias"], requires_grad=False)
            #linear layer logits/probs
            self.fc2_logits = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.fc2_logits.weight = nn.Parameter(pretrained_params["decoder.fc2_logits.weight"],requires_grad=False)
            self.fc2_logits.bias = nn.Parameter(pretrained_params["decoder.fc2_logits.bias"],requires_grad=False)
            #linear layer angles
            self.fc1_angles = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1_angles.weight = nn.Parameter(pretrained_params["decoder.fc1_angles.weight"],requires_grad=False)
            self.fc1_angles.bias = nn.Parameter(pretrained_params["decoder.fc1_angles.bias"], requires_grad=False)
            #linear layer angles means
            self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_means.weight = nn.Parameter(pretrained_params["decoder.fc2_means.weight"],requires_grad=False)
            self.fc2_means.bias = nn.Parameter(pretrained_params["decoder.fc2_means.bias"], requires_grad=False)
            #linear layer angles kappas
            self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_kappas.weight = nn.Parameter(pretrained_params["decoder.fc2_kappas.weight"],requires_grad=False)
            self.fc2_kappas.bias = nn.Parameter(pretrained_params["decoder.fc2_kappas.bias"], requires_grad=False)
            #RNN
            self.sru = SRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              bidirectional=True,
                              num_layers=2,
                              dropout=0.0)

            #TODO: Check if correct
            self.sru.rnn_lst.weight= nn.Parameter(pretrained_params["decoder$$$sru.rnn_lst.0.weight"],requires_grad=False)
            self.sru.rnn_lst.weight_c = nn.Parameter(pretrained_params["decoder$$$sru.rnn_lst.0.weight_c"],requires_grad=False)
            self.sru.rnn_lst.bias =  nn.Parameter(pretrained_params["ecoder$$$sru.rnn_lst.0.bias"],requires_grad=False)

        else:
            self.fc1_logits = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc2_logits = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.fc1_angles = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
            #Highlight: SRU Taken From https://github.com/asappresearch/sru
            self.sru = SRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)

    def forward(self, input, hidden):
        input = input.permute(1,0,2) #[max_len_seq,n_nodes,rnn_input]
        sru_output, sru_hidden = self.sru(input)  # [max_seq_len,n_nodes,2*H] | [n_layers, n_nodes, 2*H]
        sru_output = sru_output.permute(1,0,2)
        output_logits = self.fc1_logits(sru_output)
        output_logits = self.logsoftmax(self.fc2_logits(output_logits))  # [n_nodes,max_seq_len,aa_probs]
        output_angles = self.fc1_angles(sru_output)
        output_means = self.tanh(self.fc2_means(output_angles))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(output_angles))
        return output_logits,output_means,output_kappas
class RNNDecoder_Angles_Double_SRU(nn.Module):
    def __init__(self, max_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Angles_Double_SRU, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = 3
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        if pretrained_params is not None:
            #linear layer logits
            self.fc1_logits = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1_logits.weight = nn.Parameter(pretrained_params["decoder.fc1_logits.weight"],requires_grad=False)
            self.fc1_logits.bias = nn.Parameter(pretrained_params["decoder.fc1_logits.bias"], requires_grad=False)
            #linear layer logits/probs
            self.fc2_logits = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.fc2_logits.weight = nn.Parameter(pretrained_params["decoder.fc2_logits.weight"],requires_grad=False)
            self.fc2_logits.bias = nn.Parameter(pretrained_params["decoder.fc2_logits.bias"],requires_grad=False)
            #linear layer angles
            self.fc1_angles = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1_angles.weight = nn.Parameter(pretrained_params["decoder.fc1_angles.weight"],requires_grad=False)
            self.fc1_angles.bias = nn.Parameter(pretrained_params["decoder.fc1_angles.bias"], requires_grad=False)
            #linear layer angles means
            self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_means.weight = nn.Parameter(pretrained_params["decoder.fc2_means.weight"],requires_grad=False)
            self.fc2_means.bias = nn.Parameter(pretrained_params["decoder.fc2_means.bias"], requires_grad=False)
            #linear layer angles kappas
            self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_kappas.weight = nn.Parameter(pretrained_params["decoder.fc2_kappas.weight"],requires_grad=False)
            self.fc2_kappas.bias = nn.Parameter(pretrained_params["decoder.fc2_kappas.bias"], requires_grad=False)
            #RNN
            self.sru_aa = SRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              bidirectional=True,
                              num_layers=2,
                              dropout=0.0)
            self.sru_angles = SRU(input_size=self.rnn_input_size,
                                  hidden_size=self.gru_hidden_dim,
                                  bidirectional=True,
                                  num_layers=2,
                                  dropout=0.0)
            #TODO: Check if correct
            self.sru_aa.rnn_lst.weight= nn.Parameter(pretrained_params["decoder$$$sru_aa.rnn_lst.0.weight"],requires_grad=False)
            self.sru_aa.rnn_lst.weight_c = nn.Parameter(pretrained_params["decoder$$$sru_aa.rnn_lst.0.weight_c"],requires_grad=False)
            self.sru_aa.rnn_lst.bias =  nn.Parameter(pretrained_params["ecoder$$$sru_aa.rnn_lst.0.bias"],requires_grad=False)
            self.sru_angles.rnn_lst.weight = nn.Parameter(pretrained_params["decoder$$$sru_angles.rnn_lst.0.weight"],requires_grad=False)
            self.sru_angles.rnn_lst.weight_c = nn.Parameter(pretrained_params["decoder$$$sru_angles.rnn_lst.0.weight_c"],requires_grad=False)
            self.sru_angles.rnn_lst.bias = nn.Parameter(pretrained_params["ecoder$$$sru_angles.rnn_lst.0.bias"],requires_grad=False)

        else:
            self.fc1_logits = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc2_logits = nn.Linear(self.gru_hidden_dim, self.aa_prob)
            self.fc1_angles = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
            self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
            #Highlight: SRU Taken From https://github.com/asappresearch/sru
            self.sru_aa = SRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)
            self.sru_angles = SRU(input_size=self.rnn_input_size,
                                 hidden_size=self.gru_hidden_dim,
                                 bidirectional=True,
                                 num_layers=self.num_layers,
                                 dropout=0.0)

    def forward(self, input, hidden):
        input = input.permute(1,0,2) #[max_len_seq,n_nodes,rnn_input]
        sru_aa_output, sru_aa_hidden = self.sru_aa(input)  # [max_seq_len,n_nodes,2*H] | [n_layers, n_nodes, 2*H]
        sru_angles_output, sru_angles_hidden = self.sru_angles(input)  # [max_seq_len,n_nodes,2*H] | [n_layers, n_nodes, 2*H]
        sru_aa_output = sru_aa_output.permute(1,0,2)
        sru_angles_output = sru_angles_output.permute(1,0,2)
        output_logits = self.fc1_logits(sru_aa_output)
        output_logits = self.logsoftmax(self.fc2_logits((output_logits)))  # [n_nodes,max_seq_len,aa_probs]
        output_angles = self.fc1_angles(sru_angles_output)
        output_means = self.tanh(self.fc2_means(output_angles))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(output_angles))
        return output_logits,output_means,output_kappas

class RNNDecoder_Tiling_AnglesComplex(nn.Module):
    def __init__(self, max_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling_AnglesComplex, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.max_seq_len = max_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        if pretrained_params is not None:
            #common linear layer
            self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1.weight = nn.Parameter(pretrained_params["decoder.fc1.weight"],requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["decoder.fc1.bias"], requires_grad=False)
            # logits linear layer
            self.fc1_probs = nn.Linear(self.gru_hidden_dim,int((self.gru_hidden_dim)//2))
            self.fc1_probs.weight = nn.Parameter(pretrained_params["decoder.fc1_probs.weight"], requires_grad=False)
            self.fc1_probs.bias = nn.Parameter(pretrained_params["decoder.fc1_probs.bias"], requires_grad=False)
            #means linear layer
            self.fc1_means = nn.Linear(self.gru_hidden_dim,int((self.gru_hidden_dim)//2))
            self.fc1_means.weight = nn.Parameter(pretrained_params["decoder.fc1_means.weight"], requires_grad=False)
            self.fc1_means.bias = nn.Parameter(pretrained_params["decoder.fc1_means.bias"], requires_grad=False)
            #kappas linear layer
            self.fc1_kappas = nn.Linear(self.gru_hidden_dim,int((self.gru_hidden_dim)//2))
            self.fc1_kappas.weight = nn.Parameter(pretrained_params["decoder.fc1_kappas.weight"], requires_grad=False)
            self.fc1_kappas.bias = nn.Parameter(pretrained_params["decoder.fc1_kappas.bias"], requires_grad=False)
            #linear layer logits/probs
            self.fc2_probs = nn.Linear(int((self.gru_hidden_dim)//2), self.aa_prob)
            self.fc2_probs.weight = nn.Parameter(pretrained_params["decoder.fc2_probs.weight"],requires_grad=False)
            self.fc2_probs.bias = nn.Parameter(pretrained_params["decoder.fc2_probs.bias"],requires_grad=False)
            #linear layer angles means
            self.fc2_means = nn.Linear(int((self.gru_hidden_dim)//2), 2)
            self.fc2_means.weight = nn.Parameter(pretrained_params["decoder.fc2_means.weight"],requires_grad=False)
            self.fc2_means.bias = nn.Parameter(pretrained_params["decoder.fc2_means.bias"], requires_grad=False)
            #linear layer angles kappas
            self.fc2_kappas = nn.Linear(int((self.gru_hidden_dim)//2), 2)
            self.fc2_kappas.weight = nn.Parameter(pretrained_params["decoder.fc2_kappas.weight"],requires_grad=False)
            self.fc2_kappas.bias = nn.Parameter(pretrained_params["decoder.fc2_kappas.bias"], requires_grad=False)
            #RNN
            self.rnn = nn.GRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)
            #self.rnn.__setstate__(pretrained_params) #Is this working?
            self.rnn.weight_ih_l0 = nn.Parameter(pretrained_params["decoder.rnn.weight_ih_l0"],requires_grad=False)
            self.rnn.weight_hh_l0 = nn.Parameter(pretrained_params["decoder.rnn.weight_hh_l0"],requires_grad=False)
            self.rnn.bias_ih_l0 = nn.Parameter(pretrained_params["decoder.rnn.bias_ih_l0"],requires_grad=False)
            self.rnn.bias_hh_l0 = nn.Parameter(pretrained_params["decoder.rnn.bias_hh_l0"],requires_grad=False)
            self.rnn.weight_ih_l0_reverse =nn.Parameter(pretrained_params["decoder.rnn.weight_ih_l0_reverse"],requires_grad=False)
            self.rnn.weight_hh_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.weight_hh_l0_reverse"],requires_grad=False)
            self.rnn.bias_ih_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.bias_ih_l0_reverse"],requires_grad=False)
            self.rnn.bias_hh_l0_reverse = nn.Parameter(pretrained_params["decoder.rnn.bias_hh_l0_reverse"],requires_grad=False)
            #self.rnn.train(False)

        else:
            self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc1_probs = nn.Linear(self.gru_hidden_dim,int((self.gru_hidden_dim)//2))
            self.fc1_means = nn.Linear(self.gru_hidden_dim, int((self.gru_hidden_dim) // 2))
            self.fc1_kappas = nn.Linear(self.gru_hidden_dim, int((self.gru_hidden_dim) // 2))
            self.fc2_probs = nn.Linear(int((self.gru_hidden_dim) // 2), self.aa_prob)
            self.fc2_means = nn.Linear(int((self.gru_hidden_dim) // 2), 2)
            self.fc2_kappas = nn.Linear(int((self.gru_hidden_dim) // 2), 2)
            self.rnn = nn.GRU(input_size=self.rnn_input_size,
                              hidden_size=self.gru_hidden_dim,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=self.num_layers,
                              dropout=0.0)

    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,max_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output = self.fc1(rnn_output)
        output_logits = self.logsoftmax(self.fc2_probs(self.fc1_probs(output)))  # [n_nodes,max_seq_len,aa_probs]
        output_means = self.tanh(self.fc2_means(self.fc1_means(output)))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(self.fc1_kappas(output)))
        return output_logits,output_means,output_kappas

class Embed(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(Embed, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        if pretrained_params is not None:
            self.fc1 = nn.Linear(self.aa_probs, self.aa_probs)
            self.fc1.weight = nn.Parameter(pretrained_params["embed.fc1.weight"],requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["embed.fc1.bias"],requires_grad=False)
        else:
            self.fc1 = nn.Linear(self.aa_probs,self.aa_probs)
        self.logsoftmax = nn.Softmax(dim=-1)
        # self.embedding = nn.Embedding(num_embeddings=1,
        #                               embedding_dim=1,
        #                               padding_idx=None,
        #                               max_norm=None,
        #                               norm_type=2.0,
        #                               scale_grad_by_freq=False,
        #                               sparse=False,
        #                               _weight=None) #TODO: Not working with current cudann
    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        return output
class EmbedComplex(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(EmbedComplex, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        self.logsoftmax = nn.Softmax(dim=-1)

        if pretrained_params is not None:
            self.fc1 = nn.Linear(self.aa_probs, self.embedding_dim)
            self.fc1.weight = nn.Parameter(pretrained_params["embed.fc1.weight"], requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["embed.fc1.bias"], requires_grad=False)
            self.fc2 = nn.Linear(self.embedding_dim, self.aa_probs)
            self.fc2.weight = nn.Parameter(pretrained_params["embed.fc2.weight"], requires_grad=False)
            self.fc2.bias = nn.Parameter(pretrained_params["embed.fc2.bias"], requires_grad=False)
        else:
            self.fc1 = nn.Linear(self.aa_probs,self.embedding_dim)
            self.fc2 = nn.Linear(self.embedding_dim,self.aa_probs)

    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.logsoftmax(self.fc2(output))
        return output
class EmbedComplexEncoder(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(EmbedComplexEncoder, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        self.logsoftmax = nn.Softmax(dim=-1)

        if pretrained_params is not None: #TODO: Fix
            self.fc1 = nn.Linear(self.aa_probs, self.embedding_dim)
            self.fc1.weight = nn.Parameter(pretrained_params["embeddingencoder.fc1.weight"], requires_grad=False)
            self.fc1.bias = nn.Parameter(pretrained_params["embeddingencoder.fc1.bias"], requires_grad=False)
            self.fc2 = nn.Linear(self.embedding_dim, self.aa_probs)
            self.fc2.weight = nn.Parameter(pretrained_params["embeddingencoder.fc2.weight"], requires_grad=False)
            self.fc2.bias = nn.Parameter(pretrained_params["embeddingencoder.fc2.bias"], requires_grad=False)
        else:
            self.fc1 = nn.Linear(self.aa_probs,self.embedding_dim)
            self.fc2 = nn.Linear(self.embedding_dim,self.aa_probs)

    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.logsoftmax(self.fc2(output))
        return output


class GPKernel(ABC):
    @abstractmethod
    def preforward(self, t1: torch.Tensor,t2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
class SVIEngine(Engine):
    def __init__(self, *args, step_args=None, **kwargs):
        self.svi = SVI(*args, **kwargs)
        self._step_args = step_args or {}
        super(SVIEngine, self).__init__(self._update)

    def _update(self, engine, batch):
        return -engine.svi.step(batch, **self._step_args)
class OUKernel_SimulationFunctionalValuesTraits(GPKernel):
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb

    def preforward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
        sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
        lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes)
        sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise
        larger l implies that the noise should be bigger to capture big point fluctuations
        """
        first_term = self.sigma_f ** 2
        second_term = torch.exp(-t / self.lamb)
        return first_term * second_term + self.sigma_n ** 2 * torch.eye(t.shape[0])

class OUKernel_Fast(GPKernel):
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb
    def preforward(self,t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = t1.unsqueeze(1) - t2.unsqueeze(0)
        absdiff = diff.abs().sum(-1)
        return absdiff

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
        Computes a covariance matrix for each of the OU processes occurring in the latent space
        sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
        lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes)
        sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
        larger l implies that the noise should be bigger to capture big point fluctuations
        """
        #cov_b = torch.exp(-distance_matrix / _lambd) * _sigma_f ** 2 + _sigma_n + torch.eye(self.n_b*2, device=self.device) * 1e-5
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1)
        lamb = self.lamb.unsqueeze(-1).unsqueeze(-1) #self.lamb[:, None, None]
        second_term = torch.exp(-t / lamb)
        noise = torch.eye(t.shape[0]) #distributes noise/stochascity to diagonal of the covariance
        sigma_n = self.sigma_n.unsqueeze(-1).unsqueeze(-1)
        return first_term * second_term + sigma_n ** 2 * noise
class OUKernel_Fast_Sparse(GPKernel):
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb
    def preforward(self,t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = t1.unsqueeze(1) - t2.unsqueeze(0)
        absdiff = diff.abs().sum(-1)
        return absdiff
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
        sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
        lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes)
        sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise
        larger l implies that the noise should be bigger to capture big point fluctuations
        """
        first_term = self.sigma_f ** 2
        second_term = torch.exp(-t / self.lamb[:, None, None])
        return first_term[:, None, None] * second_term + self.sigma_n[:, None, None] ** 2
class VSGP(TorchDistribution):
    """
    Variational Sparse Gaussian Process distribution
    Follow this if anything is missing: https://github.com/pytorch/pytorch/blob/21c2542b6a9faafce0b6a3e1583a07b3fba9269d/torch/distributions/multivariate_normal.py
    """

    def __init__(self, kernel: GPKernel, inducing_set: torch.Tensor,
                 output_distribution_f: Callable[[torch.Tensor, torch.Tensor], dist.Distribution],
                 *, input_data: torch.Tensor, eps=1e-1):
        super().__init__()
        arg_constraints = {'loc': constraints.real_vector,
                           'covariance_matrix': constraints.positive_definite,
                           'precision_matrix': constraints.positive_definite,
                           'scale_tril': constraints.lower_cholesky}
        self.kernel = kernel
        self.inducing_set = inducing_set #selects a bunch of nodes from a Normal distribution
        self.output_distribution_f = output_distribution_f
        self.input_data = input_data #patristic distances
        self._eps = eps
        self._compute_out_dist()
        self.support = constraints.real_vector
        #batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]

    def _compute_out_dist(self):
        induced_induced = self.kernel.preforward(self.inducing_set.unsqueeze(-1),self.inducing_set.unsqueeze(-1))
        kmm = self.kernel.forward(induced_induced) #[z_dim,n_inducing_points,n_inducing_points]
        noise = torch.eye(kmm.size()[1]) * self._eps #[n_inducing_points,n_inducing_points]
        kmm = kmm + noise[None,:,:] #TODO: We need to add the noise outside, because of shape problems
        input_induced = self.kernel.preforward(self.input_data,self.inducing_set.unsqueeze(-1)) #[n_seq,n_inducing points]
        knm = self.kernel.forward(input_induced) #[n_seq,n_nodes,n_inducing_points]
        kmm_inv = kmm.inverse() #[z_dim,n_inducing_points,n_inducing_points]
        kmn = knm.transpose(-1, -2) #[z_dim,n_inducing_points,n_seq]
        input_input = self.kernel.preforward(self.input_data,self.input_data) #[n_seq,n_seq]
        knn = self.kernel.forward(input_input) ##[z_dim,n_seq,n_seq]
        #self._f_mean = knm @ kmm_inv @ self.inducing_set #[z_dim,n_seq] #TODO: What is this?
        self._f_mean = torch.zeros([self.input_data.shape[0]])

        self._f_var = knn - knm @ kmm_inv @ kmn ##[z_dim,n_seq,n_seq]
        self._out_dist = self.output_distribution_f(self._f_mean[None, :], self._f_var)

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def conjugate_update(self, other):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        return self._out_dist.sample(*args, **kwargs)

    def log_prob(self, x, *args, **kwargs):
        return self._out_dist.log_prob(x, *args, **kwargs)

    def support(self):
        """
        Returns a :class:`~torch.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
        return self.support()

def masking(Dataset):
    mask_indx = Dataset.eq(0)  # Find where is equal to 0 == gap
    Dataset_mask = torch.ones(Dataset.shape)
    Dataset_mask[mask_indx] = 0
    return Dataset_mask
def printDivisors(n) :
    i = 1
    divisors = []
    while i <= n :
        if (n % i==0) :
            divisors.append(i)
        i = i + 1
    return divisors
def intervals(parts, duration):
    part_duration = duration / parts
    return [(int(i) * part_duration, (int(i) + 1) * part_duration) for i in range(parts)]
def compute_sites_entropies(logits, node_names):
    """
    Calculate the Shannon entropy of a sequence
    logits = [n_seq, L, 21]
    observed = [n_seq,L]
    Pick the aa with the highest logit,
    logits = log(prob/1-prob)
    prob = exp(logit)/(1+exp(logit))
    entropy = prob.log(prob) per position in the sequence
    The entropy H is maximal when each of the symbols in the position has equal probability
    The entropy H is minimal when one of the symbols has probability 1 and the rest 0. H = 0"""
    #probs = torch.exp(logits)  # torch.sum(probs,dim=2) returns 1's so , it's correct

    prob = torch.exp(logits) / (1 + torch.exp(logits))
    seq_entropies = -torch.sum(prob*torch.log(prob),dim=2)

    seq_entropies = torch.cat((node_names[:,None],seq_entropies),dim=1)
    return seq_entropies
def compute_seq_probabilities(logits, observed,train=True):
    """logits = [n_seq, L, 21]
    observed = [n_seq,L]
    Pick the aa with the highest logit,
    logits = log(prob/1-prob)
    prob = exp(logit)/(1+exp(logit))"""
    #probs = torch.exp(logits)  # torch.sum(probs,dim=2) returns 1's so , it's correct
    node_names = observed[:, 0, 1]
    aminoacids = observed[:,2:,0]
    prob = torch.exp(logits) / (1 + torch.exp(logits))
    if train:
        prob_max = prob.gather(2, aminoacids[:, :, None]) #pick the probability corresponding to the observed aminoacid

    else: #for the test we "do not have" the observed sequences, so we use the highest logits as a reference
        prob_argmax = torch.argmax(prob,dim=2)
        prob_max = prob.gather(2,prob_argmax[:,:,None])
    print("min prob {}".format(torch.min(prob_max)))
    print("max prob {}".format(torch.max(prob_max)))
    #seq_probabilities = torch.sum(torch.log(prob_max), 1) #transform into log not to lose information in the product
    seq_probabilities = torch.prod(prob_max, 1)  # transform into log not to lose information in the product

    seq_probabilities = torch.cat((node_names[:,None],seq_probabilities),dim=1)
    return seq_probabilities


