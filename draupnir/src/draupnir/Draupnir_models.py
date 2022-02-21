"""
2021: aleatoryscience
Lys Sanz Moreta
Draupnir : GP prior VAE for Ancestral Sequence Resurrection
=======================
"""
import torch
import sys
from collections import defaultdict,namedtuple
from torch.distributions import constraints, transform_to
sys.path.append("./draupnir/src/draupnir")
import Draupnir_utils as DraupnirUtils
from Draupnir_models_utils import *
#from Draupnir_models_utils_bis import *
import pyro
#from tree_lstm import TreeLSTM
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
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
        self.align_seq_len = ModelLoad.align_seq_len
        self.aa_probs = ModelLoad.build_config.aa_probs
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
        self.dataset_train_blosum = ModelLoad.dataset_train_blosum
        self.variable_score = ModelLoad.variable_score
        self.internal_nodes = ModelLoad.internal_nodes
        self.batch_size = ModelLoad.build_config.batch_size
        self.plating = ModelLoad.args.plating
        self.plate_size = ModelLoad.build_config.plate_subsample_size
        self.plate_unordered = ModelLoad.plate_unordered
        self.one_hot_encoding = ModelLoad.one_hot_encoding
        if self.batch_size > 1 : # for normal batching and  batch by clade
            self.n_leaves = self.batch_size
            self.n_internal = len(self.internal_nodes)
            self.n_all = self.n_leaves + self.n_internal
        else:
            self.n_leaves = len(self.leaves_nodes)
            self.n_internal = len(self.internal_nodes)
            self.n_all = self.n_leaves + self.n_internal
        self.num_layers = 1 #TODO: Remove
        #self.h_0_GUIDE = nn.Parameter(torch.rand(self.gru_hidden_dim), requires_grad=True).to(self.device)
        # if self.pretrained_params is not None:
        #     self.h_0_MODEL = nn.Parameter(self.pretrained_params["h_0_MODEL"], requires_grad=False).to(self.device)
        # else:
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.decoder_attention = RNNAttentionDecoder(self.n_leaves, self.align_seq_len,self.aa_probs,self.gru_hidden_dim, self.rnn_input_size,self.embedding_dim, self.z_dim, self.kappa_addition)
        #self.decoder = RNNDecoder_FCL(self.align_seq_len, self.aa_probs,self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.batch_first)
        if ModelLoad.args.use_cuda:
            self.cuda()
    @abstractmethod
    def guide(self,family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum = None):
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
        """Computes a Ornstein Ulenbeck process prior over the latent space, representing the evolutionary process.
        The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances tu build a covariance matrix"""
        # Highlight; OU kernel parameters
        alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(1))
        sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.z_dim, ]).to_event(1))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
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
        #noise = 1e-15 + torch.eye(OU_covariance.shape[1])
        #https://github.com/pyro-ppl/pyro/issues/702
        #https://forum.pyro.ai/t/runtimeerror-during-cholesky-decomposition/1216/2---> fix runtime error with choleky decomposition
        #https://forum.pyro.ai/t/using-constraints-within-an-nn-module/486
        #OU_covariance = transform_to(constraints.lower_cholesky)(OU_covariance) #check that this does not affect performance
        latent_space = pyro.sample('latent_z', dist.MultivariateNormal(OU_mean, OU_covariance ).to_event(1)) #[z_dim=30,n_nodes] #+ noise[None,:,:]
        latent_space = latent_space.T
        return latent_space

    def map_sampling(self,map_estimates,patristic_matrix_full):
        "Use map sampling for leaves prediction/testing, when internal nodes are not available"
        test_indexes = (patristic_matrix_full[1:, 0][..., None] == self.internal_nodes).any(-1) #indexes of the leaves selected for testing
        latent_space_internal = map_estimates["latent_z"][:, test_indexes].T
        assert latent_space_internal.shape == (self.n_internal, self.z_dim)
        return latent_space_internal

    def conditional_sampling(self,map_estimates, patristic_matrix):
            """Conditional sampling the internal nodes given the leaves from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)
            :param map_estimates: dictionary conatining the MAP estimates for the OU process parameters
            :param patristic_matrix: full patristic matrix"""
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
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.z_dim, self.n_all, self.n_all)
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[:, internal_indexes, :]
            inverse_internal = inverse_internal[:, :, internal_indexes]  # [z_dim,n_test,n_test]
            assert inverse_internal.shape == (self.z_dim, self.n_internal, self.n_internal)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:, :, ~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.z_dim, self.n_internal, self.n_leaves)
            # Highlight: xb
            xb = map_estimates["latent_z"]  # [z_dim,n_train]
            if self.leaves_testing:
                leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1) #only the indexes of the training leaves
                xb = xb[:,leaves_indexes]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            part1 = torch.matmul(torch.linalg.inv(inverse_internal), inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal)
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.linalg.inv(inverse_internal) + 1e-6).to_event(1).sample()
            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal, self.z_dim)
            return latent_space
    def conditional_samplingMAP(self,map_estimates, patristic_matrix):
            """Conditional sampling the internal nodes given the leaves from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)
            :param map_estimates: dictionary conatining the MAP estimates for the OU process parameters
            :param patristic_matrix: full patristic matrix"""
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
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.z_dim, self.n_all, self.n_all)
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[:, internal_indexes, :]
            inverse_internal = inverse_internal[:, :, internal_indexes]  # [z_dim,n_test,n_test]
            assert inverse_internal.shape == (self.z_dim, self.n_internal, self.n_internal)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:, :, ~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.z_dim, self.n_internal, self.n_leaves)
            # Highlight: xb
            xb = map_estimates["latent_z"]  # [z_dim,n_train]
            if self.leaves_testing:
                leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1) #only the indexes of the training leaves
                xb = xb[:,leaves_indexes]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            part1 = torch.matmul(torch.linalg.inv(inverse_internal), inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal)
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.linalg.inv(inverse_internal)).to_event(1).sample()
            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal, self.z_dim)
            return OU_mean.squeeze(-1).T


class DRAUPNIRModel_classic(DRAUPNIRModelClass):
    """Implements the ordinary version of Draupnir. It receives as an input the entire leaves dataset, uses a GRU as the mapping function and blosum embeddings"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim + self.aa_probs
        self.num_layers = 1
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
    def model(self, family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum = None):
        aminoacid_sequences = family_data[:, 2:, 0]
        #batch_nodes = family_data[:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        # Highlight: GP prior over the latent space
        latent_space = self.gp_prior(patristic_matrix_sorted)

        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim] #This can maybe be done with new axis solely
        blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2):
                logits = self.decoder.forward(
                    input=latent_space,
                    hidden=decoder_hidden)
                pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_argmax=False,use_test=True,use_test2=False):
        if use_test2: #MAP estimate
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            latent_space = self.conditional_samplingMAP(map_estimates,patristic_matrix)
            n_nodes = self.n_internal #I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
        elif use_test:# Marginal posterior
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            latent_space = self.conditional_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal #I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_.shape[0], 1).reshape(latent_space_.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_ = torch.cat((latent_space_, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]

        with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2,subsample_size=n_nodes):
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

class DRAUPNIRModel_classic_no_blosum(DRAUPNIRModelClass):
    """Implements the ordinary version of Draupnir without blosum embeddings.
    It receives as an input the entire leaves dataset, uses a GRU as the mapping function WITHOUT blosum embeddings"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
    def model(self, family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum = None):
        aminoacid_sequences = family_data[:, 2:, 0]
        batch_nodes = family_data[:, 0, 1]
        # Highlight: Register GRU module
        pyro.module("decoder", self.decoder)
        # Highlight: GP prior over the latent space
        latent_space = self.gp_prior(patristic_matrix_sorted)
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2) :
                logits = self.decoder.forward(
                    input=latent_space,
                    hidden=decoder_hidden)
                pyro.sample("aa_sequences", dist.Categorical(logits=logits),
                            obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_argmax=False,use_test=True,use_test2=False):
        if use_test or use_test2:
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            latent_space = self.conditional_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal #I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)

        with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2,subsample_size=n_nodes):
            logits = self.decoder.forward(
                input=latent_space_,
                hidden=decoder_hidden)
            if use_argmax:
                #Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
                aa_sequences = torch.argmax(logits,dim=2).unsqueeze(0) #I add one dimension at the beginning to resemble 1 sample and not have to change all the plotting code
            else:
                aa_sequences = dist.Categorical(logits=logits).sample([n_samples])
        #return aa_sequences,latent_space, logits, None, None
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

class DRAUPNIRModel_classic_plating(DRAUPNIRModelClass):
    """Implements the plated version of Draupnir.
     a) It receives as an input the entire leaves dataset
     b) plates or subsamples the sequences when mapping them to the observations, no blosum embedding split
     c) uses a GRU as the mapping function.
    NOTE: The plating of the leaves nodes can be with the ordered nodes (same order as input) or random order"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim + self.aa_probs
        self.num_layers = 2
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
        self.splitted_leaves_indexes = list(torch.tensor_split(torch.arange(self.n_leaves), int(self.n_leaves / self.plate_size)) * self.num_epochs)
        if self.plate_unordered:
            self.model = self.model_unordered
        else:
            self.model = self.model_ordered
    def model_ordered(self, family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum = None):
        aminoacid_sequences = family_data[:, 2:, 0]
        batch_nodes = family_data[:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        # Highlight: GP prior over the latent space
        latent_space = self.gp_prior(patristic_matrix_sorted)
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
        blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        blosum = self.embed(blosum) #TODO: Introduce a noise variable to be able to deal with more random mutations?
        latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  #TODO: Should the subsamples are sharing the same initial hidden state?

        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2, subsample=self.splitted_leaves_indexes.pop(0)) as indx:  # Highlight: Ordered subsampling
                logits = self.decoder.forward(
                    input=latent_space[indx],
                    hidden=decoder_hidden[:,indx])
                pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences[indx])  # aa_seq = [n_nodes,align_seq_len]

    def model_unordered(self, family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum = None):
        aminoacid_sequences = family_data[:, 2:, 0]
        batch_nodes = family_data[:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        # Highlight: GP prior over the latent space
        latent_space = self.gp_prior(patristic_matrix_sorted)
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
        blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        blosum = self.embed(blosum) #TODO: Introduce a noise variable to be able to deal with more random mutations?
        latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  #TODO: Should the subsamples are sharing the same initial hidden state?

        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            with pyro.plate("plate_seq",aminoacid_sequences.shape[0],dim=-2,subsample_size=self.plate_size) as indx:#Highlight: Random subsampling
            #with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2,subsample=self.splitted_leaves_indexes.pop(0)) as indx:  # Highlight: Ordered subsampling
                logits = self.decoder.forward(
                    input=latent_space[indx],
                    hidden=decoder_hidden[:,indx])
                pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences[indx])  # aa_seq = [n_nodes,align_seq_len]



    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_argmax=False,use_test=True,use_test2=False):
        if use_test or use_test2:
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            latent_space = self.conditional_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal #I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_b = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_b.shape[0], 1).reshape(latent_space_b.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_b = torch.cat((latent_space_b, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]

        with pyro.plate("plate_len",self.align_seq_len, dim=-1):
            with pyro.plate("plate_seq",n_nodes,dim=-2,subsample_size=n_nodes) as indx:
                logits = self.decoder.forward(
                    input=latent_space_b[indx],
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

class DRAUPNIRModel_plating(DRAUPNIRModelClass):
    """Implements the plated version of Draupnir.
     a) It receives as an input the entire leaves dataset and patristic distance matrix to perform full inference over the latent space
     b) Plates or subsamples the sequences when mapping them to the observations, SPLITS the blosum embeddings accordingly.
     c) uses a GRU as the mapping function from z to aa"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim + self.aa_probs

        #self.decoder_attention = RNNAttentionDecoder(self.n_leaves, self.align_seq_len, self.aa_probs, self.gru_hidden_dim,self.rnn_input_size,self.embedding_dim, self.z_dim, self.kappa_addition)
        self.num_layers = 1
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
        self.n_splits =  int(self.n_leaves/self.plate_size)
        self.splitted_leaves_indexes = list(torch.tensor_split(torch.arange(self.n_leaves), int(self.n_leaves/self.plate_size))*self.num_epochs)


    def model(self, family_data, patristic_matrix_sorted,cladistic_matrix,subsample_split):
        aminoacid_sequences = family_data[:, 2:, 0]
        batch_nodes = family_data[:, 0, 1]
        batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        # Highlight: GP prior over the latent space
        latent_space = self.gp_prior(patristic_matrix_sorted)
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
        # blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        # blosum = self.embed(blosum) #TODO: Introduce a noise variable to be able to deal with more random mutations?
        # latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                               latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  # bidirectional #TODO: same hidden state sliced or new hidden states for every subsample
        #next(iter(self.splitted_leaves_indexes)))

        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            #with pyro.plate("plate_seq",aminoacid_sequences.shape[0],dim=-2,subsample_size=self.plate_size) as indx:#Highlight: Random subsampling
            with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2,subsample= self.splitted_leaves_indexes.pop(0))as indx:  # Highlight: Ordered subsampling
                latent_space_subsample = latent_space[indx]
                aa_frequencies = DraupnirUtils.calculate_aa_frequencies_torch(aminoacid_sequences[indx],self.aa_probs) #TODO, merge in class function
                blosum_max, blosum_weighted, variable_score = DraupnirUtils.process_blosum(self.blosum, aa_frequencies,self.align_seq_len,self.aa_probs)
                blosum_embedding = blosum_weighted.repeat(latent_space_subsample.shape[0], 1).reshape(latent_space_subsample.shape[0],
                                                                                       self.align_seq_len,
                                                                                       self.aa_probs)  # [n_nodes,max_seq,21]
                blosum_embedding = self.embed(blosum_embedding)
                latent_space_subsample = torch.cat((latent_space_subsample, blosum_embedding), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]
                logits = self.decoder.forward(
                        input=latent_space_subsample,
                        hidden=decoder_hidden[:,indx])
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences[indx]) #aa_seq = [n_nodes,align_seq_len]

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_argmax=False,use_test=True,use_test2=False):
        if use_test or use_test2:
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            latent_space = self.conditional_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal #I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_extended = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_extended.shape[0], 1).reshape(latent_space_extended.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_extended = torch.cat((latent_space_extended, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]

        with pyro.plate("plate_len",self.align_seq_len, dim=-1):
            with pyro.plate("plate_seq",n_nodes,dim=-2,subsample_size=n_nodes):
                logits = self.decoder.forward(
                    input=latent_space_extended,
                    hidden=decoder_hidden)
                if use_argmax:#Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
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


class DRAUPNIRModel_cladebatching(DRAUPNIRModelClass):
    """Implements the clade batched version of Draupnir with full latent space inference.
     a) It receives as an input a clade of the leaves dataset
     b) uses the unsplitted leaves patristic matrix for full latent space inference
     c} uses a GRU as the mapping function and blosum embeddings."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim + self.aa_probs
        #self.decoder_attention = RNNAttentionDecoder(self.n_leaves, self.align_seq_len, self.aa_probs, self.gru_hidden_dim,self.rnn_input_size,self.embedding_dim, self.z_dim, self.kappa_addition)
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
    def model(self, family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum):
        aminoacid_sequences = family_data[:, 2:, 0]
        batch_nodes = family_data[:, 0, 1]
        batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        #pyro.module("decoder_attention", self.decoder_attention)
        pyro.module("decoder", self.decoder)
        # Highlight: GP prior over the latent space of all the leaves
        latent_space = self.gp_prior(patristic_matrix_sorted)
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
        #if clade_blosum is not None: #Highlight: For clade training we only use the blosum information of that part of the alignment that includes the sequences in the clade TODO: Move outside training
        blosum = clade_blosum.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        #else: #blosum embedding is based on the entire alignment
        #blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]
        batch_latent_space = latent_space[batch_indexes] #In order to reduce the load on the GRU memory we split the latent space of the leaves by clades/batch
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                               batch_latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1), pyro.plate("plate_seq",aminoacid_sequences.shape[0],dim=-2):
            logits = self.decoder.forward(
                    input=batch_latent_space,
                    hidden=decoder_hidden)
            pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,align_seq_len]

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_argmax=False,use_test=True,use_test2=False):
        if use_test:
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            latent_space = self.conditional_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal #DO NOT REMOVE: I had to write this line because of some weird data cases (coral dataset), otherwise family_data_test.shape[0] would have sufficed
        elif use_test2:
            #latent_space = self.conditional_sampling_descendants(map_estimates,patristic_matrix)
            latent_space = self.conditional_sampling_descendants_leaves(map_estimates,patristic_matrix)
            n_nodes = self.n_internal
        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_extended = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_extended.shape[0], 1).reshape(latent_space_extended.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_extended = torch.cat((latent_space_extended, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]

        with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2):
            logits = self.decoder.forward(
                    input=latent_space_extended,
                    hidden=decoder_hidden)
            if use_argmax: #Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
                aa_sequences = torch.argmax(logits,dim=2).unsqueeze(0) #I add one dimension at the beginning to resemble 1 sample and not have to change all the plotting code
            else:
                aa_sequences = dist.Categorical(logits=logits).sample([n_samples])
        #return aa_sequences,latent_space, logits, None, None
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

class DRAUPNIRModel_leaftesting(DRAUPNIRModelClass):
    """Leaves training and testing. Train on full leave latent space (train + test), only observe the pre-selected train leaves"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim + self.aa_probs
        #self.decoder_attention = RNNAttentionDecoder(self.n_leaves, self.align_seq_len, self.aa_probs, self.gru_hidden_dim,self.rnn_input_size,self.embedding_dim, self.z_dim, self.kappa_addition)
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
    def model(self, family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum):
        aminoacid_sequences = family_data[:, 2:, 0]
        #angles = family_data[:, 2:, 1:3]
        train_nodes = family_data[:, 0, 1]
        train_indexes = (patristic_matrix_sorted[1:, 0][..., None] == train_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        # Highlight: GP prior over the latent space of all the leaves
        latent_space = self.gp_prior(patristic_matrix_sorted)
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
        blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                               latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1), pyro.plate("plate_seq",aminoacid_sequences.shape[0],dim=-2):
            logits = self.decoder.forward(
                    input=latent_space,
                    hidden=decoder_hidden)
            #Highlight: Observe only some of the leaves
            logits = logits[train_indexes]
            pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,align_seq_len]

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_argmax=False,use_test=True,use_test2=False):
        if use_test2:
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            latent_space = self.conditional_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal #DO NOT REMOVE: I had to write this line because of some weird data cases (coral dataset), otherwise family_data_test.shape[0] would have sufficed
        elif use_test:
            #latent_space = self.conditional_sampling_descendants(map_estimates,patristic_matrix)
            latent_space = self.map_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal
        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_all, self.z_dim)
            leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1)
            latent_space = latent_space[leaves_indexes]
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_extended = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_extended.shape[0], 1).reshape(latent_space_extended.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_extended = torch.cat((latent_space_extended, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]

        with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2):
            logits = self.decoder.forward(
                    input=latent_space_extended,
                    hidden=decoder_hidden)

            # batch_nodes = family_data_test[:, 0, 1]
            # batch_indexes = (patristic_matrix[1:, 0][..., None] == batch_nodes).any(-1)
            # batch_logits = logits[batch_indexes]
            if use_argmax: #Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
                aa_sequences = torch.argmax(logits,dim=2).unsqueeze(0) #I add one dimension at the beginning to resemble 1 sample and not have to change all the plotting code
            else:
                aa_sequences = dist.Categorical(logits=logits).sample([n_samples])
        #return aa_sequences,latent_space, logits, None, None
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

class DRAUPNIRModel_anglespredictions(DRAUPNIRModelClass):
    "Leaves training and testing.Predicting both ANGLES and AA sequence. Working on full or partial leaves space"
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.rnn_input_size = self.z_dim + self.aa_probs
        self.decoder = RNNDecoder_Tiling_Angles(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.rnn_input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
    def model(self, family_data, patristic_matrix_sorted,cladistic_matrix,clade_blosum):
        aminoacid_sequences = family_data[:, 2:, 0]
        angles = family_data[:, 2:, 1:3]
        angles_mask = torch.where(angles == 0., angles, 1.).type(angles.dtype) #keep as 0 the gaps and set to 1 where there is an observation
        train_nodes = family_data[:, 0, 1]
        train_indexes = (patristic_matrix_sorted[1:, 0][..., None] == train_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        # Highlight: GP prior over the latent space of all the leaves
        latent_space = self.gp_prior(patristic_matrix_sorted)
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention

        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]

        blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                               latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1), pyro.plate("plate_seq",aminoacid_sequences.shape[0],dim=-2):
            logits,means,kappas = self.decoder.forward(
                input=latent_space,
                hidden=decoder_hidden)
            logits = logits[train_indexes]
            means = means[train_indexes]
            kappas = kappas[train_indexes]
            pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,align_seq_len]
            pyro.sample("phi",dist.VonMises(loc = means[:,:,0],concentration = kappas[:,:,0]).mask(angles_mask), obs=angles[:,:,0])
            pyro.sample("psi",dist.VonMises(loc = means[:,:,1],concentration = kappas[:,:,1]).mask(angles_mask), obs=angles[:,:,1])

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,cladistic_matrix,use_argmax=False,use_test=True,use_test2=False):
        if use_test:
            #latent_space = self.conditional_sampling_descendants(map_estimates,patristic_matrix)
            latent_space = self.map_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal
        elif use_test2:
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            latent_space = self.conditional_sampling(map_estimates,patristic_matrix)
            n_nodes = self.n_internal #DO NOT REMOVE: I had to write this line because of some weird data cases (coral dataset), otherwise family_data_test.shape[0] would have sufficed
        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_all, self.z_dim)
            leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1)
            latent_space = latent_space[leaves_indexes]
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves

        #decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                               latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()  # Contains 2 hidden states in 1, to be processed by different GRU/SRUs

        latent_space_extended = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_extended.shape[0], 1).reshape(latent_space_extended.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_extended = torch.cat((latent_space_extended, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]

        with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2):

            logits,means,kappas = self.decoder.forward(
                input=latent_space_extended,
                hidden=decoder_hidden)

            if use_argmax: #Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
                aa_sequences = torch.argmax(logits,dim=2).unsqueeze(0) #I add one dimension at the beginning to resemble 1 sample and not have to change all the plotting code
            else:
                aa_sequences = dist.Categorical(logits=logits).sample([n_samples])
            phis = dist.VonMises(loc=means[:,:,0],concentration = kappas[:,:,0]).sample([n_samples])
            psis = dist.VonMises(loc=means[:,:,1],concentration = kappas[:,:,1]).sample([n_samples])

        sampling_out = SamplingOutput(aa_sequences=aa_sequences.detach(),
                                      latent_space=latent_space.detach(),
                                      logits=logits.detach(),
                                      phis=phis.detach(),
                                      psis=psis.detach(),
                                      mean_phi = means[:,:,0].detach(),
                                      mean_psi = means[:,:,1].detach(),
                                      kappa_phi = kappas[:,:,0].detach(),
                                      kappa_psi = kappas[:,:,1].detach())
        return sampling_out

