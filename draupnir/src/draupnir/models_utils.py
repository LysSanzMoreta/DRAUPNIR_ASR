"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
# TORCH
import torch.nn as nn
import torch
import  torch.nn.functional as F
from torch.distributions import constraints
import math
from ignite.engine import Engine, Events
from abc import ABC, abstractmethod
from typing import Callable
#Pyro
from pyro.infer import SVI
from pyro import distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution


class RNNEncoder(nn.Module):
    def __init__(self, align_seq_len,
                 aa_prob,n_leaves,
                 gru_hidden_dim,
                 z_dim,
                 input_size,
                 kappa_addition,
                 num_layers,
                 pretrained_params):
        super(RNNEncoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.nndim = int(self.gru_hidden_dim/2)
        self.fc1 = nn.Linear(self.gru_hidden_dim, self.nndim)
        self.linear_means = nn.Linear(self.nndim, self.z_dim)
        self.linear_std = nn.Linear(self.nndim, self.z_dim)

        #self.layernorm = nn.LayerNorm(self.input_size)
        self.softplus = nn.Softplus()

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

        #todo: necessary? maybe
        #self.init_gru_bias()
    def init_gru_bias(self): #some funky suggestion by llm
        for name, param in self.rnn.named_parameters(): #initialization if the gru bias to avoid dominance
            if "bias_ih_l0" in name:
                param.data[self.gru_hidden_dim:2 * self.gru_hidden_dim] = -1.0  # reset gate bias


    def forward(self, input, hidden):

        #input = self.layernorm(input)
        rnn_hidden_states, rnn_final_bidirectional = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim*2] | [num_layers*2,n_nodes,gru_dim]
        # print("Encoder rnn states")
        # print(rnn_final_bidirectional.var(dim=0).mean())  # should not be tiny
        # print(rnn_hidden_states.var(dim=0).mean())  # should not be tiny
        forward_out_r,backward_out_r = rnn_hidden_states[:,:,:self.gru_hidden_dim],rnn_hidden_states[:,:,self.gru_hidden_dim:]

        rnn_hidden_states = forward_out_r + backward_out_r #original
        rnn_final_forward_backward_sum = rnn_hidden_states[:,-1] #original takes the last state of the forward and the first state of the backward

        #rnn_final_forward_backward_sum = forward_out_r[:,-1] + backward_out_r[:,0] #pick the last state forward and the first state from backwards


        rnn_final_hidden_state = self.fc1(rnn_final_forward_backward_sum)
        z_loc = self.linear_means(rnn_final_hidden_state)
        z_scale = self.softplus(self.linear_std(rnn_final_hidden_state))

        return  {"z_loc":z_loc,
                 "z_scale":z_scale,
                 "rnn_final_bidirectional":rnn_final_bidirectional,
                 "rnn_final_forward_backward_sum":rnn_final_forward_backward_sum.unsqueeze(0), #not transformed
                 "rnn_hidden_states":rnn_hidden_states,
                 "rnn_final_hidden_state":rnn_final_hidden_state}

        #return output_means,output_std

class FCLEncoder(nn.Module):
    def __init__(self, align_seq_len,aa_prob,n_leaves,gru_hidden_dim, z_dim,input_size, num_layers):
        super(FCLEncoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(self.input_size,self.gru_hidden_dim) #todo: put in sequential module
        self.fc2 = nn.Linear(self.gru_hidden_dim,self.gru_hidden_dim)
        self.linear_means = nn.Linear(self.gru_hidden_dim, self.z_dim)
        self.linear_std = nn.Linear(self.gru_hidden_dim, self.z_dim)
        self.layernorm = nn.LayerNorm(self.input_size)
        self.softplus = nn.Softplus()


        #todo: necessary? maybe
        #self.init_gru_bias()
    def init_gru_bias(self):
        for name, param in self.rnn.named_parameters(): #initialization if the gru bias to avoid dominance
            if "bias_ih_l0" in name:
                param.data[self.gru_hidden_dim:2 * self.gru_hidden_dim] = -1.0  # reset gate bias


    def forward(self, input, hidden):

        #input = self.layernorm(input)


        output = self.fc2(self.fc1(input))
        z_loc = self.linear_means(output)
        z_scale = self.softplus(self.linear_std(output))

        return  {"z_loc":z_loc,
                 "z_scale":z_scale}

        #return output_means,output_std

class xLSTMEncoder(nn.Module):
    def __init__(self,max_len,input_size,z_dim):
        super(xLSTMEncoder, self).__init__()
        from xlstm import (
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            sLSTMBlockConfig,
            sLSTMLayerConfig,
            FeedForwardConfig,
        )

        self.input_size = input_size
        self.z_dim = z_dim
        self.num_heads = 3

        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,  # seems to be unaffected
                    qkv_proj_blocksize=self.num_heads,  # set to num_heads
                    num_heads=self.num_heads,
                    proj_factor=2.0,
                    embedding_dim=-1,
                    bias=False,
                    dropout=0.0,
                    context_length=-1,
                    round_proj_up_to_multiple_of=90,  # inner_embedding_dim, must be divisible by num_heads
                    round_proj_up_dim_up=True
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    embedding_dim=-1,
                    num_heads=self.num_heads,
                    # this must divide the hidden size, is not yet supported by all versions in this directory
                    conv1d_kernel_size=4,
                    group_norm_weight=True,
                    dropout=0,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3,
                                              act_fn="gelu",
                                              embedding_dim=-1,
                                              dropout=0,
                                              bias=False,
                                              ff_type="ffn_gated"),
            ),
            context_length=max_len,
            num_blocks=4,  # does not seem to matter
            embedding_dim=self.input_size,
            slstm_at=[1],
            add_post_blocks_norm=True,
            bias=False,
            dropout=0.0,

        )

        self.xlstm_stack = xLSTMBlockStack(self.cfg)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.linear_means = nn.Linear(self.input_size, self.z_dim)
        self.linear_std = nn.Linear(self.input_size, self.z_dim)


        #TODO: need to manually move to cuda?
    def forward(self,input):

        embedding = self.xlstm_stack(input)

        # print(embedding.max())
        # print(embedding.min())
        # print(torch.isinf(embedding).any())
        # print(torch.isnan(embedding).any())
        # print("-----------0-------------")
        # print(embedding[:,0])
        # print("-----------1-------------")
        # print(embedding[:,1])
        # print("-----------2-------------")
        # print(embedding[:,2])
        # print("-----------5-------------")
        # print(embedding[:,5])

        latent = embedding.mean(axis=1)
        latent = self.fc1(latent)
        z_loc = self.linear_means(latent)
        z_scale = self.logsoftmax(latent)

        return {"embeddings": embedding,
                "z_loc": z_loc,
                "z_scale": z_scale
                }
class xLSTMDecoder(nn.Module):
    def __init__(self,max_len,input_size,z_dim,output_size):
        super(xLSTMDecoder, self).__init__()
        from xlstm import (
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            sLSTMBlockConfig,
            sLSTMLayerConfig,
            FeedForwardConfig,
        )

        self.input_size = input_size
        self.output_size = output_size
        self.z_dim = z_dim
        self.num_heads = 3

        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,  # seems to be unaffected
                    qkv_proj_blocksize=self.num_heads,  # set to num_heads
                    num_heads=self.num_heads,
                    proj_factor=2.0,
                    embedding_dim=-1,
                    bias=False,
                    dropout=0.0,
                    context_length=-1,
                    round_proj_up_to_multiple_of=90,  # inner_embedding_dim, must be divisible by num_heads
                    round_proj_up_dim_up=True
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    embedding_dim=-1,
                    num_heads=self.num_heads,
                    # this must divide the hidden size, is not yet supported by all versions in this directory
                    conv1d_kernel_size=4,
                    group_norm_weight=True,
                    dropout=0,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3,
                                              act_fn="gelu",
                                              embedding_dim=-1,
                                              dropout=0,
                                              bias=False,
                                              ff_type="ffn_gated"),
            ),
            context_length=max_len,
            num_blocks=4,  # does not seem to matter
            embedding_dim=self.input_size,
            slstm_at=[1],
            add_post_blocks_norm=True,
            bias=False,
            dropout=0.0,

        )

        self.xlstm_stack = xLSTMBlockStack(self.cfg)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.linear_probs = nn.Linear(self.input_size, self.output_size)

    def forward(self,input):

        embedding = self.xlstm_stack(input)
        #
        # print(embedding.max())
        # print(embedding.min())
        # print(torch.isinf(embedding).any())
        # print(torch.isnan(embedding).any())

        output_logits = self.logsoftmax(self.linear_probs(self.fc1(embedding))) #todo: layernorm

        return output_logits


class RNNDecoder_Tiling(nn.Module):
    def __init__(self,
                 align_seq_len,
                 aa_probs,
                 gru_hidden_dim,
                 z_dim,
                 input_size,
                 kappa_addition,
                 num_layers,
                 pretrained_params):
        super(RNNDecoder_Tiling, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_probs = aa_probs
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.layernorm = nn.LayerNorm(self.input_size)
        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_probs)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)
        #todo: necessary? does not seem to hurt

        #self.init_gru_bias()

    def init_gru_bias(self):
        for name, param in self.rnn.named_parameters(): #initialization if the gru bias to avoid dominance
            if "bias_ih_l0" in name:
                param.data[self.gru_hidden_dim:2 * self.gru_hidden_dim] = -1.0  # reset gate bias


    # def forward(self, input, hidden): #old function for RNNDecoder_Tiling_new
    #     rnn_hidden_states, rnn_final_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
    #     forward_hidden_states = rnn_hidden_states[:, :, :self.gru_hidden_dim]
    #     backward_hidden_states = rnn_hidden_states[:, :, self.gru_hidden_dim:]
    #     rnn_hidden_states = forward_hidden_states + backward_hidden_states
    #     #rnn_final_hidden = rnn_hidden_states[:,-1]
    #
    #     rnn_final_hidden = self.fc1(rnn_hidden_states)
    #     output_logits = self.logsoftmax(self.linear_probs(rnn_final_hidden))  # [n_nodes,align_seq_len,aa_probs]
    #     #output_logits = self.logsoftmax(self.linear_probs(self.fc1(rnn_final_hidden)))  # [n_nodes,align_seq_len,aa_probs]
    #     return output_logits

    def forward(self, input, hidden):
        """One-shot, non-autoregressive sequence generation"""
        #input = self.layernorm(input) #added extra

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim] #rnn_out is not expressive? whereas the hidden states are

        # print("Decoder rnn states")
        # print(rnn_output.var(dim=0).mean())  # should not be tiny
        # print(rnn_hidden.var(dim=0).mean())  # should not be tiny
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        #rnn_output = self.layernorm(rnn_output) #worsens training, not necessary, since all the sequences an in the same "scale"
        output_logits = self.logsoftmax(self.linear_probs(self.fc1(rnn_output)))  # [n_nodes,align_seq_len,aa_probs]

        return output_logits
class RNNDecoder_Tiling_Angles(nn.Module):
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling_Angles, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = 4
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.fc2_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
        self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
        self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output = self.fc1(rnn_output)
        output_logits = self.logsoftmax(self.fc2_probs((output)))  # [n_nodes,align_seq_len,aa_probs]
        output_means = self.tanh(self.fc2_means(output))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(output))
        return output_logits,output_means,output_kappas
class RNNDecoder_Tiling_AnglesComplex(nn.Module):
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling_AnglesComplex, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.fc1_probs = nn.Linear(self.gru_hidden_dim,int((self.gru_hidden_dim)//2))
        self.fc1_means = nn.Linear(self.gru_hidden_dim, int((self.gru_hidden_dim) // 2))
        self.fc1_kappas = nn.Linear(self.gru_hidden_dim, int((self.gru_hidden_dim) // 2))
        self.fc2_probs = nn.Linear(int((self.gru_hidden_dim) // 2), self.aa_prob)
        self.fc2_means = nn.Linear(int((self.gru_hidden_dim) // 2), 2)
        self.fc2_kappas = nn.Linear(int((self.gru_hidden_dim) // 2), 2)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output = self.fc1(rnn_output)
        output_logits = self.logsoftmax(self.fc2_probs(self.fc1_probs(output)))  # [n_nodes,align_seq_len,aa_probs]
        output_means = self.tanh(self.fc2_means(self.fc1_means(output)))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(self.fc1_kappas(output)))
        return output_logits,output_means,output_kappas
class RNNDecoder_TeacherForcing(nn.Module): #Highlight: faster learning for some reason
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_TeacherForcing, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.method = "normal"
        if self.method == "tf":
            self.layernorm = nn.LayerNorm(self.gru_hidden_dim)
            self.fc1 = nn.Linear(self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc2 = nn.Linear(self.gru_hidden_dim,self.input_size)
            self.linear_probs = nn.Linear(self.input_size, self.aa_prob)
        else:
            #self.layernorm = nn.LayerNorm(self.gru_hidden_dim)

            self.fc1 = nn.Linear( self.gru_hidden_dim, self.input_size ) #+ 2*self.gru_hidden_dim
            self.fc2 = nn.Linear(self.input_size , self.gru_hidden_dim)
            self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
        #self.fc_hidden = nn.Linear(2*self.input_size,2*self.input_size)


        self.rnn = nn.GRU(input_size=self.input_size, # + 2*self.gru_hidden_dim,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=False,
                          num_layers=self.num_layers)


    def forward_teacher_forcing(self, input, hidden): #todo: delete, apparently, teacher forcing is only used during sampling, not training

        """
        :param hidden: encoder hidden states (except in the first round)

        :return

        output_logits: [N,L,feat_dim], where feat_dim == aa_probs
        """
        #hidden = torch.tanh(self.fc_hidden(hidden)) # need to scale the encoder hidden state

        if hidden.ndim < 3:
            hidden = hidden.unsqueeze(0)
        token = input[:,0].unsqueeze(1)
        Bs, L = input.shape[0], input.shape[1]
        output_logits = torch.zeros((Bs,L,self.aa_prob))
        for t in range(L):
            rnn_out_t, rnn_hidden = self.rnn(token, hidden)  # [n_seqs,1, 2* gru_dim] | [n_directions, n_seqs, gru_dim]
            rnn_out_t = self.fc2(self.fc1(rnn_out_t))
            rnn_out_t = self.layernorm(rnn_out_t)
            output_logit_t = self.logsoftmax(self.linear_probs(rnn_out_t))
            output_logits[:,t] = output_logit_t.squeeze(1)
            if t+1 < L:
                token = input[:, t + 1].unsqueeze(1)  # todo: try teacher forcing
                #token = input[:,t].unsqueeze(1) if torch.rand(1) < self.teacher_forcing_ratio else rnn_out_t
        #output_logits = self.layernorm(output_logits) # [n_nodes,align_seq_len,aa_probs]
        return output_logits

    def forward_normal(self,input,hidden,mode="train"):
        """
        :input: embeddings, not direct tokens
        """

        if mode == "train":
            #hidden = torch.tanh(self.fc_hidden(hidden))

            encoder_hidden_states, decoder_hidden = hidden["encoder_hidden_states"], hidden["decoder"]
            #input = torch.cat([encoder_hidden_states,input],dim=-1)
            rnn_output, rnn_hidden = self.rnn(input, decoder_hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
            #rnn_output = self.layernorm(rnn_output) #worsens training, not necessary, since all the sequences an in the same "scale"
            rnn_output = self.fc1(rnn_output)
            output_logits = self.logsoftmax(self.linear_probs(self.fc2(rnn_output)))  # [n_nodes,align_seq_len,aa_probs]

        elif mode == "sample":
            encoder_hidden_states, decoder_hidden = hidden["encoder_hidden_states"], hidden["decoder"]
            input = torch.cat([encoder_hidden_states, input], dim=-1)
            Bs, L, feat_dim = input.shape

            #input_t = input[:, 0].unsqueeze(1) # "first token"
            input_t = torch.zeros([Bs,1, feat_dim]) # "start token"

            output_logits = torch.zeros((Bs, L, self.aa_prob))
            for t in range(L):
                rnn_output_t, rnn_hidden_t = self.rnn(input_t, decoder_hidden)
                # #rnn_output_t = self.layernorm(rnn_output_t)
                rnn_output_t = self.fc1(rnn_output_t)

                if 0 < t  < L :
                    input_t = rnn_output_t + input[:,t].unsqueeze(1)# just the rnn_output_t is not working

                else:
                    input_t = rnn_output_t
                output_logit_t = self.logsoftmax(self.linear_probs(self.fc2(rnn_output_t)))
                output_logits[:, t] = output_logit_t.squeeze(1)



            #output_logits = output_logits[:,1:]



        return output_logits

    def forward(self,input,hidden,mode="train"):
        if self.method == "tf":
            return self.forward_teacher_forcing(input,hidden)

        else:
            return self.forward_normal(input,hidden,mode)
class Embed(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(Embed, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.aa_probs,self.aa_probs)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        return output
class EmbedComplex(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(EmbedComplex, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.aa_probs,self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim,self.aa_probs)
    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.softmax(self.fc2(output))
        return output
class EmbedComplexEncoder(nn.Module):
    def __init__(self,input_dim,embedding_dim,out_dim):
        super(EmbedComplexEncoder, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.input_dim,self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim,self.out_dim)

    def forward(self,input):

        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.softmax(self.fc2(output))

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
    """ Kernel that computes the covariance matrix for a z Ornstein Ulenbeck processes. As stated in Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
    :param tensor sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
    :param tensor lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes),larger l implies that the noise should be bigger to capture big point fluctuations
    :param tensor sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
    **References:**
    "Ancestral Inference from Functional Data: Statistical Methods and Numerical Examples"
    """
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb

    def preforward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        first_term = self.sigma_f ** 2
        second_term = torch.exp(-t / self.lamb)
        return first_term * second_term + self.sigma_n ** 2 * torch.eye(t.shape[0])

class OUKernel_Fast(GPKernel):
    """ Kernel that computes the covariance matrix for a z Ornstein Ulenbeck processes. As stated in Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
    :param tensor sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
    :param tensor lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes),larger l implies that the noise should be bigger to capture big point fluctuations
    :param tensor sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
    **References:**
    "Ancestral Inference from Functional Data: Statistical Methods and Numerical Examples"
    """
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb
    def preforward(self,t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Not used function"""

        return torch.zeros((1,))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        #cov_b = torch.exp(-distance_matrix / _lambd) * _sigma_f ** 2 + _sigma_n + torch.eye(self.n_b*2, device=self.device) * 1e-5
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1) #[:,None,None]
        lamb = self.lamb.unsqueeze(-1).unsqueeze(-1) #self.lamb[:, None, None]
        second_term = torch.exp(-t / lamb)
        noise = torch.eye(t.shape[0]) #distributes noise/stochascity to diagonal of the covariance
        sigma_n = self.sigma_n.unsqueeze(-1).unsqueeze(-1)
        return first_term * second_term + sigma_n ** 2 * noise

class OUKernel_Fast_Sparse(GPKernel):
    """ Kernel that computes the covariance matrix for a z Ornstein Ulenbeck processes, in this case for a sparse Gaussian process. As stated in Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
    :param tensor sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
    :param tensor lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes),larger l implies that the noise should be bigger to capture big point fluctuations
    :param tensor sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
    """
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb
    def preforward(self,t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = t1.unsqueeze(1) - t2.unsqueeze(0)
        absdiff = diff.abs().sum(-1)
        return absdiff
    def forward(self, t: torch.Tensor) -> torch.Tensor:
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

class PositionalEncodings(nn.Module):
    def __init__(self,max_seq_len,feat_dim,base=1000, type="sinusoidal"):
        super(PositionalEncodings, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.max_seq_len = max_seq_len
        self.feat_dim = feat_dim
        self.base = base
        self.positions_idx = torch.arange(0, self.max_seq_len)[None, :]  # [1,L]
        self.dimensions_idx = torch.arange(0, self.feat_dim // 2 )
        self.type = type

    def align(self,tensor, axes, ndim=None): #TODO: Do I need this? Does not seem to do anything
        """Expand dimensions
        axes：
        ndim：
        """
        assert len(axes) == tensor.ndim
        assert ndim or min(axes) >= 0
        ndim = ndim or max(axes) + 1
        indices = [None] * ndim
        for i in axes:
            indices[i] = slice(None)
        return tensor[indices]
    def sinusoidal_encodings(self):
        """
        Implements trigonometric or sinusoidal positional encodings. Identical results to those in the annotated transformer

        1) Calculation of frequencies: feat_dim=10, base=10000
        MethodA : torch.pow(self.base , -2 * torch.arange(0,self.feat_dim//2) / self.feat_dim)
        [0,1,2,3,4] -> divide by dim ->  [0,0,1,0.2,0.3,0.4] -> [0,-0.2,-0.4,-0.6,-0.8] -> power -> [10000⁰, 10000^{-0.2}, 10000^{-0.4},10000^{-0.6}, 10000^{-0.8}] (already inverted)
        MethodB:  1/ (self.base**(torch.arange(0,self.feat_dim,2)/self.feat_dim))
        [0,2,4,6,8] -> divide by dim -> [0,0.2,0.4,0.6,0.8] --> power -> [10000⁰, 10000^{0.2}, 10000^{0.4},10000^{0.6}, 10000^{0.8}] -> invert -> [1/10000⁰, 1/10000^{0.2}, ....]

        2) Vector multiplication via Einstein summation:

        y = a1x1 + a2x2 + ... + anxn ->    y = sum(aixi) -> ii

        #Example1: Selects the diagonal values (00,11,22) (the trace) and sums them so the result is 15 (1 + 5 + 9)
        # >>> torch.einsum("ii",torch.tensor([[1,2,3],[4,5,6],[7,8,9]]))
        # 15
        #Example2: 1D vector multiplication: c00 = a0*b0; c01 = a0*b1 ....
        # >>> a = torch.tensor([3,4,5])
        # >>> b = torch.tensor([6,7,8])
        # >>> torch.einsum("i,j->ij",a,b)
        tensor([[18, 21, 24],
                [24, 28, 32],
                [30, 35, 40]])
        # The above is equivalent to torch.einsum("...,i->...i",a,b) and to torch.matmul(a[:,None],b[None,:])

        3) Stacking the sine and cosine results in an overlapping manner:

        Given the frequencies for a single feature vector of size 3 (feat_dim) at time step 0 of the sequence
        frequencies  = [\theta1, \theta2, \theta3]
        sine_frequencies = [sin\theta1, sin\theta2, sin\theta3]
        cosine_frequencies = [cos\theta1, cos\theta2, cos\theta3]

        stack = torch.stack([sine_frequencies,cosine_frequencies],dim=-2)
        stack
             [[sin\theta1,cos\theta1],
              [sin\theta2,cos\theta2],
              [sin\theta3,cos\theta3]]
        stack.flatten(-2)
             [sin\theta1,cos\theta1,sin\theta2,cos\theta2,sin\theta3,cos\theta3]
        when we index the odd positions of the flattened stack latter, we will get the values for the cosine and fof the even, the sine

        SOURCE: https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L845

        returns:
            :param sinusoidal_embeddings of shape [L,feat_dim*2]

        """

        frequencies= torch.pow(self.base , -2 * self.dimensions_idx / self.feat_dim) #frequencies or angle rates #[feat_dim]
        #frequencies = 1/ (self.base**(torch.arange(0,self.feat_dim,2)/self.feat_dim)) #equivalent code, but needs one more operation
        frequencies = torch.einsum('...,d->...d', self.positions_idx,frequencies) #1D vector multiplication [L,feat_dim] #equivalent to torch.einsum('i,j->ij') or torch.matmul(a[:,None],b[None,:])
        #Highlight: The upcoming code is a bit of a cumbersome one and I am not sure why they do not just keep the cosine and sine values separated always,
        # and avoid indexing. I do not think they are ever used together...
        sinusoidal_embeddings = torch.stack([torch.sin(frequencies), torch.cos(frequencies)], axis=-1) #[1,L,feat_dim//2,2]
        sinusoidal_embeddings = torch.flatten(sinusoidal_embeddings, -2) #[1,L,feat_dim]

        return sinusoidal_embeddings
    def apply_rotation(self,sinusoidal_embedding, inputs):
        """
        Implements the necessary steps to prepare the tensors for rotation as in Equation 34 in the paper https://arxiv.org/pdf/2104.09864
        1) Split the hidden dimensions from the inputs (key and query) into chunks of 2

        2) Rotation in 2D (x \in R^2 for the feature dimension)

        W = [[W00,W01],       X = [X0,X1]  ----->  X' = W@X = [[W00.X0 + W01.X1], = [X0' , X1']
             [W10,W11]]                                    [W10.X0 + W10.X1]]
        R = [[cosm\theta, -sinm\theta], ------>  R@X' = [[cosm\theta1.X0', -sinm\theta1.X0'], ---------->Split up ----------> R@X' = [[cosm\theta1] *[[x0], + [[sinm\theta1] *[[-x1],
             [sinm\theta, cosm\theta]]                 [sinm\theta1.X1', cosm\theta1.X1']]                                          [cosm\theta1]] [x1]]     [sinm\theta1]   [x0]]

        3) Rotation in n-dimensions

        R@X' = [[cosm\theta1], *[[x0], + [[sinm\theta1], *[[-x1],
                [cosm\theta1],  [x1],     [sinm\theta1],   [x0],
                [cosm\theta2],  [x2],     [sinm\theta2],   [x3],
                [cosm\theta2]]  [x3]]     [sinm\theta2]]   [x2]]

        Implements adjacent rope rank, similar to https://github.com/Tongjilibo/bert4torch/blob/ce644b9cefa72801a4a23366cb2cd9b895511951/bert4torch/layers/position_encoding.py#L170-L256C6
        return
            :param inputs of shape [N,L,...,feat_dim], where L is the max length. Inputs can be the keys and queries
        """
        ndim = inputs[0].ndim
        sinusoidal_embedding = self.align(sinusoidal_embedding,[0, 1, -1], ndim) #[1,L,featdim]
        cos_positions = torch.repeat_interleave(sinusoidal_embedding[...,1::2],2,-1) #We need to duplicate the frequencies so that we can perform the rotation in nD (\theta1,\theta1,\theta2,\theta2) #repeats the last dimension of the tensor such that [0,1,2,3,4,5] -> [0,0,1,1,2,2,3,3]
        sine_positions = torch.repeat_interleave(sinusoidal_embedding[...,::2],2,-1)

        #Highlight: from https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py#L359
        rotated_outputs = []
        for x in inputs:
            tensor2 = torch.stack([-x[..., 1::2], x[..., ::2]], ndim) #Highlight: need to use ndim, no 2
            tensor2 = torch.reshape(tensor2, x.shape) #these 2 lines turn -tensor2- into [-x2,x1,-x4,x3, ....]
            # x1,x2 = x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:] #This comes from the bert2torch implementation and it is not correct in this setting perhaps it needs a different context
            # tensor2 = torch.cat([-x2,x1], dim=x1.ndim - 1)
            rotated_outputs.append(x * cos_positions + tensor2 * sine_positions) #Equation 34

        return rotated_outputs
    def forward(self,input):

        sinusoidal_embeddings = self.sinusoidal_encodings()  # independent of the input content except for the maxlen and featdim
        if self.type == "sinusoidal":

            self.register_buffer("positional_encoding", sinusoidal_embeddings)
            input = input + sinusoidal_embeddings[:, : input.size(1)].requires_grad_(False)

            return input

        elif self.type == "rotary":
            # key = self.query_fc(input)
            # query = self.key_fc(input)
            key = input
            query = input
            #values = input

            kw, qw = self.apply_rotation(sinusoidal_embeddings, [key, query]) #TODO: WARNING : Switched qw,kw

            return kw,qw

class TransformerEncoder(nn.Module):
    def __init__(self,input_dim,align_seq_len,output_dim):
        super(TransformerEncoder, self).__init__()
        self.enable_flash = True
        self.dropout_rate = 0
        self.input_dim = input_dim
        self.output_dim = output_dim #it is not a bug, in this case we keep it the same at the
        self.align_seq_len = align_seq_len
        self.num_heads = 1
        self.use_bias = False
        self.tril_mask = True
        self.softplus = nn.Softplus()
        self.key_fc = nn.Linear(self.input_dim,self.align_seq_len * self.num_heads,
                                  bias=self.use_bias,
                                  )
        self.context_fc = nn.Linear(self.input_dim, self.output_dim * self.num_heads,
                                        bias=self.use_bias,
                                        )
        self.output_fc = nn.Linear(self.output_dim * self.num_heads, self.output_dim * self.num_heads,
                                        bias=self.use_bias,
                                        )
        self.output_fc_loc = nn.Linear(self.output_dim * self.num_heads,self.output_dim * self.num_heads,
                                  bias=self.use_bias,
                                  )
        self.output_fc_scale = nn.Linear(self.output_dim * self.num_heads, self.output_dim * self.num_heads,
                                        bias=self.use_bias,
                                        )
    def sequence_masking(self,x, mask=None, value=0, axis=None, bias=None, return_mask=False):
        """为序列条件mask的函数
        mask: 形如(batch_size, seq_len)的bool矩阵；
        value: mask部分要被替换成的值，可以是'-inf'或'inf'；
        axis: 序列所在轴，默认为1；
        bias: 额外的偏置项，或者附加的mask；
        return_mask: 是否同时返回对齐后的mask。
        """
        xndim = x.ndim
        if not (mask is None and bias is None):
            if mask is None:
                if bias.dtype == 'bool':
                    mask = bias
                    x = torch.where(mask, x, value)
                else:
                    x = x + bias
            else:
                if axis is None:
                    axes = [1]
                elif isinstance(axis, list):
                    axes = axis
                else:
                    axes = [axis]
                axes = [axis if axis >= 0 else xndim + axis for axis in axes]
                if mask.dtype != 'bool':
                    mask = mask.bool()

                full_mask = self.align(mask, [0, axes[0]], xndim)
                for axis in axes[1:]:
                    full_mask = full_mask & self.align(mask, [0, axis], xndim)

                mask = full_mask
                if bias is None:
                    x = torch.where(mask, x, value)
                elif bias.dtype == 'bool':
                    mask = mask & bias
                    x = torch.where(mask, x, value)
                else:
                    x = torch.where(mask, x + bias, value)

        if return_mask:
            return x, mask
        else:
            return x
    def attention_normalize(self, a:torch.Tensor, dim:float=-1, method:str='softmax'):
        """ Normalize or bound the attention values between 0 and 1
        methods:
            softmax；
            squared_relu: from the Gated Attention Unit with a GLU(Gated Linear Unit) https://arxiv.org/abs/2202.10447 ；
            softmax_plus: Uses entropy invariance scaling for datasets using longer lengths  https://kexue.fm/archives/8823 。
        """
        if method == 'softmax':
            return F.softmax(a, dim=dim)
        else:
            mask = (a > -1e11).float()
            l = torch.maximum(torch.sum(mask, dim=dim, keepdims=True), torch.tensor(1).to(mask))
            if method == 'squared_relu':
                return F.relu(a) ** 2 / l
            elif method == 'softmax_plus':
                return F.softmax(a * torch.log(l) / torch.log(torch.tensor(512.0)).to(mask), dim=dim)
        return a
    def attention(self,query,keys,values,mask):
        att_scores = torch.einsum('bmd,bnd->bmn', query, keys) / self.output_dim ** 0.5 #attention scores/logits(softmax)
        att_scores = self.attention_normalize(att_scores,-1,method="softmax") #[nseqs,L,L]


        #Original code is for seq2seq https://github.com/bojone/bert4keras/blob/2072f06dd410ea885a9c6850ba539effce50b22b/bert4keras/layers.py#L1485-L1487
        # bias = self.query_fc(input) / 2
        # att_scores = att_scores[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None] #in the original code the bias is divided in 2 parts because bias[:,::2,None] belongs to the query and bisas[:,1::2,:,None] belongs to the keys

        #Method2: https://github.com/Tongjilibo/bert4torch/blob/ce644b9cefa72801a4a23366cb2cd9b895511951/bert4torch/layers/global_point.py#L90-L93
        bias_input = self.key_fc(keys)  # [..., heads*feat_dim] #in the original the inputs are 1 big array?
        bias2 = torch.stack(torch.chunk(bias_input, self.num_heads, dim=-1), dim=-2).transpose(1,2) / 2  # [btz, heads, seq_len, seq_len*num_heads]

        logits = att_scores.unsqueeze(1) + bias2  # [batch_size, num_heads, seq_len, seq_len]

        if self.tril_mask:
            tril_mask = torch.triu(torch.ones_like(att_scores[0]), 0)
            tril_mask = tril_mask.bool()
        else:
            tril_mask = None

        att_scores = self.sequence_masking(att_scores, mask, -torch.inf, [2, 3], tril_mask) #TODO: understand and simplify?

        hidden_states = torch.einsum('bij, bjd -> bid', att_scores, values) #[N,L,feat_dim/hidden_dim?]


        hidden_states = self.context_fc(hidden_states) #https://github.com/Tongjilibo/bert4torch/blob/ce644b9cefa72801a4a23366cb2cd9b895511951/bert4torch/layers/attention.py#L725C9-L725C91
        #TODO: What to do with the output when num_heads > 2

        #TODO: Average? I need z_mean and z_scale for the encoder and logits for the decoder? wft
        context_vector = hidden_states.mean(1)
        context_vector = self.output_fc(context_vector)
        z_loc = self.output_fc_loc(context_vector)
        z_scale = self.softplus(self.output_fc_scale(context_vector))

        return {"attention_scores":att_scores,
                "hidden_states": hidden_states,
                "context_vector":context_vector, #TODO: Can be used as logits? [nseqs, alignlen, feat_dim]
                "attention_logits":logits,
                "z_loc":z_loc, #TODO: One loc and scale per position!!! requires changing the distribution
                "z_scale":z_scale}
    def forward(self,query,key,values,mask):
        return self.attention(query,key,values,mask)

class TransformerEncoder2(nn.Module):
    def __init__(self,input_dim,align_seq_len,output_dim):
        super(TransformerEncoder2, self).__init__()
        self.enable_flash = True
        self.dropout_rate = 0
        self.input_dim = input_dim
        self.output_dim = output_dim #it is not a bug, in this case we keep it the same at the
        self.align_seq_len = align_seq_len
        self.num_heads = 1
        self.use_bias = True
        self.softplus = nn.Softplus()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=2)
        self.norm_layer = nn.LayerNorm(self.input_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=1, norm=self.norm_layer)
        self.hidden_states_fc = nn.Linear(self.input_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )

        self.output_fc = nn.Linear(self.output_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )
        self.output_fc_loc = nn.Linear(self.output_dim ,self.output_dim ,
                                  bias=self.use_bias,
                                  )
        self.output_fc_scale = nn.Linear(self.output_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )
    def forward(self,input,mask):

        hidden_states = self.encoder(input)
        hidden_states = self.hidden_states_fc(hidden_states)
        context_vector = hidden_states.mean(axis=1)

        context_vector = self.output_fc(context_vector)
        z_loc = self.output_fc_loc(context_vector)
        z_scale = self.softplus(self.output_fc_scale(context_vector))


        return {"context_vector":context_vector,
                "hidden_states":hidden_states, #TODO: Can be used as logits? [nseqs, alignlen, feat_dim]
                "z_loc":z_loc,
                "z_scale":z_scale}

class TransformerEncoder3(nn.Module):
    def __init__(self,input_dim,adapted_input_dim, align_seq_len,output_dim):
        """
        input_dim is the embedding size (i.e aa probs)
        adapted_input_dim is the corrected input size for when the number is odd

        """
        super(TransformerEncoder3, self).__init__()

        self.dropout_rate = 0
        self.input_dim = input_dim
        self.adapted_input_dim = adapted_input_dim
        self.output_dim = output_dim

        self.align_seq_len = align_seq_len
        self.num_heads = 1
        self.use_bias = True
        self.softplus = nn.Softplus()
        self.positional_embeddings = PositionalEncodings(self.align_seq_len,self.adapted_input_dim,base=1000, type="sinusoidal")
        self.layernorm = nn.LayerNorm(self.adapted_input_dim)  # todo: not sure yet
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.adapted_input_dim, nhead=2) #includes residual connections
        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=1, norm=self.layernorm)
        self.hidden_states_fc = nn.Linear(self.adapted_input_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )
        self.augmented_output_dim = int(self.output_dim*2)
        self.output_fc = nn.Linear(self.output_dim, self.augmented_output_dim,
                                        bias=self.use_bias,
                                        )
        self.output_fc_loc = nn.Linear(self.augmented_output_dim ,self.output_dim,
                                  bias=self.use_bias,
                                  )
        self.output_fc_scale = nn.Linear(self.augmented_output_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )
    def forward(self,input,mask):

        #todo: masking & more heads?

        input = self.positional_embeddings(input) # simple summation, this returns 21 + 1 dimensions, otherwise the transformer does not work
        hidden_states = self.encoder(input)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.hidden_states_fc(hidden_states) # i have projected them to 30 perhaps project to 30)

        #context_vector = hidden_states.mean(axis=1) # mean pooling

        context_vector = hidden_states[:,0,:] # get the CLS token (the one with most information)

        context_vector = self.output_fc(context_vector)
        z_loc = self.output_fc_loc(context_vector)
        z_scale = self.softplus(self.output_fc_scale(context_vector))




        return {"context_vector":context_vector,
                "hidden_states":hidden_states,
                "z_loc":z_loc,
                "z_scale":z_scale}

class TransformerDecoder(nn.Module):
    def __init__(self,input_dim_l, input_dim_r,align_seq_len,hidden_dim,output_dim):
        super(TransformerDecoder, self).__init__()
        self.num_heads = 1
        self.num_decoder_layers = 1
        self.input_dim_l = input_dim_l
        self.input_dim_r = input_dim_r
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.align_seq_len = align_seq_len
        self.positional_embeddings = PositionalEncodings(self.align_seq_len, self.hidden_dim, base=1000,
                                                         type="sinusoidal")
        self.input_fc = nn.Linear(self.input_dim_l,self.hidden_dim)
        self.latent_fc = nn.Linear(self.input_dim_r,self.hidden_dim)
        self.layernorm = nn.LayerNorm(self.hidden_dim)  # todo: not sure yet
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.num_heads),
            num_layers= self.num_decoder_layers
        )

        self.output_fc = nn.Linear(self.hidden_dim,self.output_dim)
        self.logsoftmax = nn.LogSoftmax()
    def forward(self,input,latent):
        """
        :param torch.tensor tgt_seq : [nseq,L,featdim]
        :param torch.tensor latent: [nseq,L,zdim]
        """
        latent = self.latent_fc(latent)
        latent = self.positional_embeddings(latent)
        latent = self.layernorm(latent)
        input = self.input_fc(input)
        input = self.positional_embeddings(input)
        input = self.layernorm(input)



        decoded = self.decoder(tgt=input, memory=latent)
        logits = self.logsoftmax(self.output_fc(decoded))

        return logits


def masking(dataset):
    """Creating a mask for the gaps (0) in the data set"""
    mask_indx = dataset.eq(0)  # Find where is equal to 0 == gap
    dataset_mask = torch.ones(dataset.shape)
    dataset_mask[mask_indx] = 0
    return dataset_mask
def print_divisors(n) :
    """Calculates the number of divisors of a number
    :param int n: number"""
    i = 1
    divisors = []
    while i <= n :
        if (n % i==0) :
            divisors.append(i)
        i = i + 1
    return divisors
def intervals(parts, duration):
    """Compose a list of intervals on which a number is divided """
    part_duration = int(duration / parts)

    return [(int(i) * part_duration, (int(i) + 1) * part_duration) for i in range(parts)]
def compute_sites_entropies(logits, node_names):
    """
    Calculate the Shannon entropy of a sequence
    :param tensor logits = [n_seq, L, 21]
    :param tensor node_names: tensor with the nodes tree level order indexes ("names")
    observed = [n_seq,L]
    Pick the aa with the highest logit,
    logits = log(prob/1-prob)
    prob = exp(logit)/(1+exp(logit))
    entropy = prob.log(prob) per position in the sequence
    The entropy H is maximal when each of the symbols in the position has equal probability
    The entropy H is minimal when one of the symbols has probability 1 and the rest 0. H = 0"""
    #probs = torch.exp(logits)  # torch.sum(probs,dim=2) returns 1's so , it's correct

    prob_softmax = torch.nn.Softmax(dim=-1)(logits)

    #prob_sigmoid = torch.exp(logits) / (1 + torch.exp(logits)) #kind of sigmoid function
    seq_entropies = -torch.sum(prob_softmax*torch.log(prob_softmax),dim=2)

    seq_entropies = torch.cat((node_names[:,None],seq_entropies),dim=1)
    node_names_2d = node_names[:,None].tile((logits.shape[2])).reshape(len(node_names),1,logits.shape[2])

    prob_softmax = torch.concatenate((node_names_2d,prob_softmax),dim=1)

    return seq_entropies, prob_softmax
def compute_seq_probabilities(logits, observed,train=True):
    """Compute the sequence probabilities (prob = exp(logit)/(1+exp(logit))) from the logits
    :param tensor logits: log(prob/1-prob), [n_seq, L, 21]
    :param tensor observed = [n_seq,L]
    """
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


