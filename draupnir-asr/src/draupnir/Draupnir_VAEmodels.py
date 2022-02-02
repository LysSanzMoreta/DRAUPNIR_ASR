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
from Draupnir_models_utils_bis import *
import pyro
from tree_lstm import TreeLSTM
from pyro import poutine
from pyro.ops.indexing import Vindex
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
        self.decoder_attention = RNNAttentionDecoder(self.n_leaves, self.max_seq_len,self.aa_prob,self.gru_hidden_dim, self.rnn_input_size,self.embedding_dim, self.z_dim, self.kappa_addition)
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
    def gp_prior(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Radia basis Function-like kernel that uses the patristic distances"
        # Highlight; OU kernel parameters
        alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(1))
        sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.z_dim, ]).to_event(1))  # rate of mean reversion/selection strength---> signal variance
        sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand_by([self.z_dim, ]).to_event(1))  # Gaussian noise
        lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.z_dim, ]).to_event(1))  # characteristic length-scale
        # Highlight: Sample the latent space from MultivariateNormal with GP prior on covariance
        patristic_matrix = patristic_matrix_sorted[1:, 1:]
        OU_covariance = OUKernel_Fast(sigma_f, sigma_n, lambd).forward(patristic_matrix)
        OU_mean = torch.zeros((patristic_matrix.shape[0],)).unsqueeze(0)
        if self.leaves_testing:
            assert OU_covariance.shape == (self.z_dim, self.n_all, self.n_all)
            assert OU_mean.shape == (1, self.n_all)
        else:
            assert OU_covariance.shape == (self.z_dim, self.n_leaves, self.n_leaves)
            assert OU_mean.shape == (1,self.n_leaves)
        noise = 1e-15*torch.eye(OU_covariance.shape[1])
        latent_space = pyro.sample('latent_z', dist.MultivariateNormal(OU_mean, OU_covariance + noise[None,:,:]).to_event(1)) #[z_dim=30,n_nodes]
        latent_space = latent_space.T
        return latent_space

    def map_sampling(self,map_estimates,patristic_matrix_full):
        "Use map sampling for leaves prediction/testing, when internal nodes are not available"
        test_indexes = (patristic_matrix_full[1:, 0][..., None] == self.internal_nodes).any(-1) #indexes of the leaves selected for testing
        latent_space_internal = map_estimates["latent_z"][:, test_indexes].T
        assert latent_space_internal.shape == (self.n_internal, self.z_dim)
        return latent_space_internal

    def conditional_sampling_descendants(self,map_estimates,patristic_matrix_full):
        "Conditionally sample the internal nodes based only on the descendant internal nodes and leaves"
        sigma_f = map_estimates["sigma_f"]
        sigma_n = map_estimates["sigma_n"]
        lambd = map_estimates["lambd"]
        OU = OUKernel_Fast(sigma_f, sigma_n, lambd)
        OU_covariance_full = OU.forward(patristic_matrix_full[1:, 1:])
        Inverse_full = torch.linalg.inv(OU_covariance_full)
        assert Inverse_full.shape == (self.z_dim, self.n_all, self.n_all)
        internal_latent_spaces = dict.fromkeys(self.internal_nodes.tolist())
        for ancestor, descendants in sorted(self.descendants_dict.items()):
            ancestor = torch.tensor(ancestor)
            descendants_internal = torch.tensor(descendants["internal"])
            descendants_leaves = torch.tensor(descendants["leaves"])
            internal_indexes = (patristic_matrix_full[1:, 0][..., None] == descendants_internal).any(-1)
            leaves_indexes = (patristic_matrix_full[1:, 0][..., None] == descendants_leaves).any(-1)
            leaves_indexes_xb = (self.leaves_nodes[..., None] == descendants_leaves).any(-1) #self.leaves nodes has the same node order as the latent space from the leaves
            Inverse_ancestor = Inverse_full[:, internal_indexes, :]
            Inverse_ancestor = Inverse_ancestor[:, :, internal_indexes]
            OU_mean_internal = torch.zeros((descendants_internal.shape[0],))
            Inverse_ancestor_leaves = Inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            Inverse_ancestor_leaves = Inverse_ancestor_leaves[:, :, leaves_indexes]  # [z_dim,n_test,n_train]
            assert Inverse_ancestor_leaves.shape == (self.z_dim, descendants_internal.shape[0], descendants_leaves.shape[0])
            xb = map_estimates["latent_z"][:, leaves_indexes_xb]
            assert xb.shape == (self.z_dim,descendants_leaves.shape[0])
            OU_mean_leaves = torch.zeros((descendants_leaves.shape[0],))
            part1 = torch.matmul(torch.linalg.inv(Inverse_ancestor),Inverse_ancestor_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, descendants_internal.shape[0])
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.linalg.inv(Inverse_ancestor)).to_event(1).sample()
            assert latent_space.shape == (self.z_dim, descendants_internal.shape[0])
            internal_latent_spaces[ancestor.item()] = latent_space[:,0].unsqueeze(-1) #the first value corresponds to the ancestor we are interested in

        internal_latent_spaces = dict(sorted(internal_latent_spaces.items()))  # luckily all the time everything is sorted in ascending order
        latent_space = torch.cat(list(internal_latent_spaces.values()), dim=1).T
        assert latent_space.shape == (self.n_internal, self.z_dim)
        return latent_space

    def conditional_sampling_descendants_leaves(self,map_estimates,patristic_matrix_full):
        "Conditionally sample the internal nodes based only on the descendant leaves and in all the internal nodes"
        sigma_f = map_estimates["sigma_f"]
        sigma_n = map_estimates["sigma_n"]
        lambd = map_estimates["lambd"]
        OU = OUKernel_Fast(sigma_f, sigma_n, lambd)
        OU_covariance_full = OU.forward(patristic_matrix_full[1:, 1:])
        Inverse_full = torch.linalg.inv(OU_covariance_full)
        assert Inverse_full.shape == (self.z_dim, self.n_all, self.n_all)
        internal_latent_spaces = dict.fromkeys(self.internal_nodes.tolist())
        for ancestor, descendants in sorted(self.descendants_dict.items()):
            ancestor = torch.tensor(ancestor)
            #descendants_internal = torch.tensor(descendants["internal"])
            descendants_leaves = torch.tensor(descendants["leaves"])
            internal_indexes = (patristic_matrix_full[1:, 0][..., None] == self.internal_nodes).any(-1)
            leaves_indexes = (patristic_matrix_full[1:, 0][..., None] == descendants_leaves).any(-1)
            leaves_indexes_xb = (self.leaves_nodes[..., None] == descendants_leaves).any(-1) #self.leaves nodes has the same node order as the latent space from the leaves
            Inverse_ancestor = Inverse_full[:, internal_indexes, :]
            Inverse_ancestor = Inverse_ancestor[:, :, internal_indexes]
            OU_mean_internal = torch.zeros((self.n_internal,))
            Inverse_ancestor_leaves = Inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            Inverse_ancestor_leaves = Inverse_ancestor_leaves[:, :, leaves_indexes]  # [z_dim,n_test,n_train]
            assert Inverse_ancestor_leaves.shape == (self.z_dim, self.n_internal, descendants_leaves.shape[0])
            xb = map_estimates["latent_z"][:, leaves_indexes_xb]
            assert xb.shape == (self.z_dim,descendants_leaves.shape[0])
            OU_mean_leaves = torch.zeros((descendants_leaves.shape[0],))
            part1 = torch.matmul(torch.linalg.inv(Inverse_ancestor),Inverse_ancestor_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.internal_nodes.shape[0])
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.linalg.inv(Inverse_ancestor)).to_event(1).sample()
            assert latent_space.shape == (self.z_dim, self.n_internal)
            internal_latent_spaces[ancestor.item()] = latent_space[:,0].unsqueeze(-1) #the first value corresponds to the ancestor we are interested in

        internal_latent_spaces = dict(sorted(internal_latent_spaces.items()))  # luckily all the time everything is sorted in ascending order
        latent_space = torch.cat(list(internal_latent_spaces.values()), dim=1).T
        assert latent_space.shape == (self.n_internal, self.z_dim)
        return latent_space

    def conditional_sampling_progressive(self,map_estimates,patristic_matrix_full):
        "Conditional sampling each ancestor on their exact respective children"
        #Highlight: Calculate the inverse covariance for the entire space
        sigma_f = map_estimates["sigma_f"]
        sigma_n = map_estimates["sigma_n"]
        lambd = map_estimates["lambd"]
        OU = OUKernel_Fast(sigma_f, sigma_n, lambd)
        OU_covariance_full = OU.forward(patristic_matrix_full[1:,1:])
        Inverse_full = torch.linalg.inv(OU_covariance_full)
        assert Inverse_full.shape == (self.z_dim,self.n_all,self.n_all)
        internal_latent_spaces = dict.fromkeys(self.internal_nodes.tolist())
        for ancestor, children in reversed(sorted(self.children_dict.items())): #First the most recent common ancestors of the leaves
            if ancestor is not np.nan:
                ancestor = torch.tensor(ancestor)
                children = torch.tensor(children)
                ancestor_indexes = (patristic_matrix_full[1:, 0][..., None] == ancestor).any(-1)
                children_indexes = (patristic_matrix_full[1:, 0][..., None] == children).any(-1)
                Inverse_ancestor = Inverse_full[:, ancestor_indexes, :]
                Inverse_ancestor = Inverse_ancestor[:, :, ancestor_indexes]
                OU_mean_ancestor = torch.zeros((torch.numel(ancestor),))
                Inverse_ancestor_children = Inverse_full[:,ancestor_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
                Inverse_ancestor_children = Inverse_ancestor_children[:, :, children_indexes]  # [z_dim,n_test,n_train]
                assert Inverse_ancestor_children.shape == (self.z_dim, torch.numel(ancestor), children.shape[0])
                children_leaves_idx = (self.leaves_nodes[..., None] == children).any(-1)
                check = children_leaves_idx.int().sum()
                if check.item() == children.shape[0]: #all are leaves
                    children_leaves_idx = (self.leaves_nodes[..., None] == children).any(-1)  # should be ordered
                    xb = map_estimates["latent_z"][:,children_leaves_idx]
                elif 0 < check.item() < children.shape[0]: #some are internal nodes and some are leaves
                    xb_leaf = map_estimates["latent_z"][:,children_leaves_idx]
                    xb_internal = [internal_latent_spaces[child.item()] for child in children if child.item() in internal_latent_spaces.keys()] #should only find one child, the internal one
                    xb = torch.cat([xb_leaf,torch.cat(xb_internal,dim=0)],dim=1) #general solution with 2 cats, in case is a non binary tree
                else: #all internal nodes
                    xb = [internal_latent_spaces[child.item()] for child in children]#find the keys of the children indexes and concatenate them
                    xb = torch.cat(xb, dim=1)
                OU_mean_children = torch.zeros((children.shape[0],))
                part1 = torch.matmul(torch.linalg.inv(Inverse_ancestor),Inverse_ancestor_children)  # [z_dim,n_test,n_train]
                part2 = xb - OU_mean_children[None, :]  # [z_dim,n_train]
                OU_mean = OU_mean_ancestor[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
                assert OU_mean.squeeze(-1).shape == (self.z_dim, torch.numel(ancestor))
                latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1),torch.linalg.inv(Inverse_ancestor)).to_event(1).sample()
                assert latent_space.shape == (self.z_dim,torch.numel(ancestor))
                internal_latent_spaces[ancestor.item()] = latent_space
        internal_latent_spaces = dict(sorted(internal_latent_spaces.items())) #luckily all the time everything is sorted in ascending order
        latent_space = torch.cat(list(internal_latent_spaces.values()),dim=1).T
        assert latent_space.shape == (self.n_internal,self.z_dim)
        return latent_space

    def conditional_sampling(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""
            sigma_f = map_estimates["sigma_f"]
            sigma_n = map_estimates["sigma_n"]
            lambd = map_estimates["lambd"]
            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes).any(-1)

            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_full = patristic_matrix[1:, 1:]
            assert patristic_matrix_full.shape == (self.n_all, self.n_all), "Remember to use the entire/full patristic matrix for conditional sampling!"
            OU = OUKernel_Fast(sigma_f, sigma_n, lambd)
            OU_covariance_full = OU.forward(patristic_matrix_full)
            OU_covariance_full = OU_covariance_full[:,]
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            Inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert Inverse_full.shape == (self.z_dim, self.n_all, self.n_all)
            # Highlight: B.49 Λ−1aa
            Inverse_internal = Inverse_full[:, internal_indexes, :]
            Inverse_internal = Inverse_internal[:, :, internal_indexes]  # [z_dim,n_test,n_test]
            assert Inverse_internal.shape == (self.z_dim, self.n_internal, self.n_internal)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal,))  # [n_internal,]
            # Highlight: Λab
            Inverse_internal_leaves = Inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            Inverse_internal_leaves = Inverse_internal_leaves[:, :, ~internal_indexes]  # [z_dim,n_test,n_train]
            assert Inverse_internal_leaves.shape == (self.z_dim, self.n_internal, self.n_leaves)
            # Highlight: xb
            xb = map_estimates["latent_z"]  # [z_dim,n_train]
            if self.leaves_testing:
                leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1) #only the indexes of the training leaves
                xb = xb[:,leaves_indexes]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            part1 = torch.matmul(torch.linalg.inv(Inverse_internal), Inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal)
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.linalg.inv(Inverse_internal)).to_event(1).sample()
            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal, self.z_dim)
            return latent_space

class DRAUPNIRModel_VAE(DRAUPNIRModelClass):
    "Blosum weighted average embedding. Training on leaves, testing on internal nodes, no batching"
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim + self.aa_prob
        self.decoder_attention = RNNAttentionDecoder(self.n_leaves, self.max_seq_len, self.aa_prob, self.gru_hidden_dim,self.rnn_input_size,
                                                     self.embedding_dim, self.z_dim, self.kappa_addition)
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
        # Highlight: GP prior over the latent space
        #latent_space = self.gp_prior(patristic_matrix_sorted)
        covariance = torch.eye(self.n_leaves).repeat(self.z_dim,1,1)
        mean = torch.zeros(self.z_dim,self.n_leaves)
        latent_space = pyro.sample("latent_z",dist.MultivariateNormal(mean, covariance)).T
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
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
        sampling_out = SamplingOutput(aa_sequences=aa_sequences,
                                      latent_space=latent_space,
                                      logits=logits,
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None)

        return sampling_out
