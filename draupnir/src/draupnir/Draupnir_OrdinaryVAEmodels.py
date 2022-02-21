"""
2021: aleatoryscience
Lys Sanz Moreta
Draupnir : GP prior VAE for Ancestral Sequence Resurrection
=======================
"""
import torch
from collections import defaultdict,namedtuple

import Draupnir_utils
from Draupnir_models_utils import *
import pyro
import pyro.distributions as dist
from pyro.util import ignore_jit_warnings
try:
    from torch_geometric.data import Data
except:
    pass
SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi"])
class DRAUPNIRModelClass(nn.Module):
    def __init__(self, ModelLoad):
        super(DRAUPNIRModelClass, self).__init__()
        self.num_epochs = ModelLoad.args.num_epochs
        self.gru_hidden_dim = ModelLoad.gru_hidden_dim
        self.embedding_dim = ModelLoad.args.embedding_dim #blosum embedding dim
        self.position_embedding_dim= ModelLoad.args.position_embedding_dim
        self.pretrained_params = ModelLoad.pretrained_params
        self.z_dim = ModelLoad.z_dim
        self.leaves_nodes = ModelLoad.leaves_nodes
        self.n_tree_levels = ModelLoad.n_tree_levels
        self.max_seq_len = ModelLoad.max_seq_len
        self.aa_prob = ModelLoad.build_config.aa_prob
        self.edge_info = ModelLoad.graph_coo #for the gnn + gru hybrid model
        self.nodes_representations_array = ModelLoad.nodes_representations_array
        self.dgl_graph = ModelLoad.dgl_graph
        self.children_dict = ModelLoad.children_dict
        self.closest_leaves_dict = ModelLoad.closest_leaves_dict
        self.descendants_dict = ModelLoad.descendants_dict
        self.clades_dict_all = ModelLoad.clades_dict_all
        self.max_indel_size = ModelLoad.args.max_indel_size
        self.rnn_input_size = self.z_dim
        self.use_attention = False
        self.batch_first = True
        self.leaves_testing = ModelLoad.leaves_testing
        self.batch_by_clade = ModelLoad.args.batch_by_clade
        self.device = ModelLoad.device
        self.kappa_addition = ModelLoad.args.kappa_addition
        self.aa_frequencies = ModelLoad.aa_frequencies
        self.blosum = ModelLoad.blosum
        self.blosum_max = ModelLoad.blosum_max
        self.blosum_weighted = ModelLoad.blosum_weighted
        self.variable_score = ModelLoad.variable_score
        self.internal_nodes = ModelLoad.internal_nodes
        self.batch_size = ModelLoad.build_config.batch_size
        self.plate_size = ModelLoad.build_config.plate_subsample_size
        if self.batch_size > 1 : # for normal batching and  batch by clade
            self.n_leaves = self.batch_size
            self.n_internal = len(self.internal_nodes)
            self.n_all = self.n_leaves + self.n_internal
        else:
            self.n_leaves = len(self.leaves_nodes)
            self.n_internal = len(self.internal_nodes)
            self.n_all = self.n_leaves + self.n_internal
        self.num_layers = 1 #TODO: Remove
        self.h_0_MODEL = nn.Parameter(torch.rand(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.decoder = RNNDecoder_FCL(self.max_seq_len, self.aa_prob,self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.batch_first)
        if ModelLoad.args.use_cuda:
            self.cuda()
    @abstractmethod
    def guide(self,family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum):
        raise NotImplementedError

    @abstractmethod
    def model(self, family_data,patristic_matrix,cladistic_matrix,clade_blosum):
        raise NotImplementedError

    @abstractmethod
    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_test=True):
        raise NotImplementedError
    @abstractmethod
    def get_class(self):
        full_name = self.__class__
        name = str(full_name).split(".")[-1].replace("'>","")
        return name

class DRAUPNIRModel_VAE(DRAUPNIRModelClass):
    "Blosum weighted average embedding. Training on leaves, testing on internal nodes, no batching"
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim + self.aa_prob
        self.decoder = RNNDecoder_Tiling(self.max_seq_len, self.aa_prob, self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_prob,self.embedding_dim, self.pretrained_params)

    def model(self, family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum = None):
        aminoacid_sequences = family_data[:, 2:, 0]
        batch_nodes = family_data[:, 0, 1]
        batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder_attention", self.decoder_attention)
        pyro.module("decoder", self.decoder)
        # Highlight: No prior over the latent space
        #latent_space = self.gp_prior(patristic_matrix_sorted)
        covariance = torch.eye(self.n_leaves).repeat(self.z_dim,1,1)
        mean = torch.zeros(self.z_dim,self.n_leaves)
        latent_space = pyro.sample("latent_z",dist.MultivariateNormal(mean, covariance)).T
        # Highlight: MAP the latent space to logits using GRU
        latent_space = latent_space.repeat(1,self.max_seq_len).reshape(latent_space.shape[0],self.max_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]

        blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.max_seq_len,self.aa_prob) #[n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,max_seq_len,z_dim + 21]
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2, subsample_size=self.plate_size) as indx:
                logits = self.decoder.forward(
                    input=latent_space[indx],
                    hidden=decoder_hidden[:, indx])
                pyro.sample("aa_sequences", dist.Categorical(logits=logits),
                            obs=aminoacid_sequences[indx])  # aa_seq = [n_nodes,max_seq_len]

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_argmax=False,use_test=True,use_test2=False):

        latent_space = map_estimates["latent_z"].T
        assert latent_space.shape == (self.n_leaves, self.z_dim)
        n_nodes = self.n_leaves

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.max_seq_len).reshape(n_nodes,self.max_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_.shape[0], 1).reshape(latent_space_.shape[0], self.max_seq_len,self.aa_prob)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_ = torch.cat((latent_space_, blosum), dim=2)  # [n_nodes,max_seq_len,z_dim + 21]

        with pyro.plate("plate_len",self.max_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2,subsample_size=n_nodes):
            logits = self.decoder.forward(
                input=latent_space_,
                hidden=decoder_hidden)
            if use_argmax:
                #Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
                aa_sequences = torch.argmax(logits,dim=2).unsqueeze(0) #I add one dimension at the beginning to resemble 1 sample and not have to change all the plotting code
            else:
                aa_sequences = dist.Categorical(logits=logits).sample([n_samples])
        sampling_out = SamplingOutput(aa_sequences=aa_sequences.detach(),
                                      latent_space=latent_space.detach(),
                                      logits=logits.detach(),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None)

        return sampling_out
