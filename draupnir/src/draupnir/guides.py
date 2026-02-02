"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
from types import MethodType
from abc import abstractmethod
import pyro
from pyro.contrib.easyguide import EasyGuide
from pyro.nn import PyroParam
import torch.nn as nn
import torch
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from draupnir.models_utils import *
import draupnir.models_utils as DraupnirModelsUtils



class DRAUPNIRGUIDES(EasyGuide):

    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        super(DRAUPNIRGUIDES, self).__init__(draupnir_model)
        self.guide_type = ModelLoad.args.select_guide
        self.draupnir = Draupnir
        self.args = ModelLoad.args
        self.encoder_input_size = self.draupnir.aa_probs
        self.dataset_train_blosum = self.draupnir.dataset_train_blosum
        self.batch_size = self.draupnir.batch_size
        self.batch_by_clade = self.draupnir.batch_by_clade
        #self.layernorm = nn.LayerNorm(self.draupnir.z_dim) #todo: should be embedding dim

        if self.draupnir.pretrained_params is not None:
            self.h_0_GUIDE = nn.Parameter(self.draupnir.pretrained_params["h_0_GUIDE"], requires_grad=True).to(self.draupnir.device)
        else:
            self.h_0_GUIDE = nn.Parameter(torch.randn(self.draupnir.gru_hidden_dim), requires_grad=True).to(self.draupnir.device)

        self.alpha = PyroParam(dist.HalfNormal(torch.tensor([1.0])).sample([3]),constraint=constraints.positive,event_dim=0) #constraint=constraints.interval(0., 10.)--->TODO:Event dimension??
        self.sigma_n = PyroParam(dist.HalfNormal(torch.tensor([1.0])).sample([self.draupnir.z_dim]),constraint=constraints.positive,event_dim=0)
        self.sigma_f = PyroParam(dist.HalfNormal(torch.tensor([1.0])).sample([self.draupnir.z_dim]),constraint=constraints.positive,event_dim=0)
        self.lambd = PyroParam(dist.HalfNormal(torch.tensor([1.0])).sample([self.draupnir.z_dim]),constraint=constraints.positive,event_dim=0)

        # self.r = PyroParam(dist.Gamma(torch.Tensor([2]),torch.Tensor([2])).sample([1]))
        # self.alpha = PyroParam(dist.Exponential(torch.Tensor([1])).sample([1]))

        if self.draupnir.plating:
            self.encoder_splitted_leaves_indexes = list(torch.tensor_split(torch.arange(self.draupnir.n_leaves), int(self.draupnir.n_leaves / self.draupnir.plate_size)) * self.draupnir.num_epochs)

    def get_class(self):
        full_name = self.__class__
        name = str(full_name).split(".")[-1].replace("'>","")
        return name

    def guide(self, datasets, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param patristic_matrix: matrix of patristic distances (branch lengths) between the nodes in the tree
        :param cladistic_matrix: matrix of cladistic distances between the nodes in the tree
        :param data_blosum : data encoded with blosum vectors
        :param batch_blosum : weighted average of blosum scores per column alignment for a batch of sequences"""

        raise NotImplementedError


class DRAUPNIRGuides_classic(DRAUPNIRGUIDES):
    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        DRAUPNIRGUIDES.__init__(self,draupnir_model,ModelLoad, Draupnir)

        self.encoder = RNNEncoder(align_seq_len=self.draupnir.align_seq_len,
                                  aa_prob=self.draupnir.aa_probs,
                                  n_leaves=self.draupnir.n_leaves,
                                  gru_hidden_dim=self.draupnir.gru_hidden_dim,
                                  z_dim=self.draupnir.z_dim,
                                  input_size=self.encoder_input_size,
                                  kappa_addition=self.draupnir.kappa_addition,
                                  num_layers=self.draupnir.num_layers,
                                  pretrained_params=self.draupnir.pretrained_params)
        self.embeddingencoder = EmbedComplexEncoder(input_dim=self.draupnir.aa_probs,
                                                    embedding_dim=self.draupnir.embedding_dim,
                                                    out_dim=self.draupnir.aa_probs)

    def guide(self, datasets, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param patristic_matrix: matrix of patristic distances (branch lengths) between the nodes in the tree
        :param cladistic_matrix: matrix of cladistic distances between the nodes in the tree
        :param data_blosum : data encoded with blosum vectors
        :param batch_blosum : weighted average of blosum scores per column alignment for a batch of sequences"""


        if self.batch_size == None or self.batch_size > 1:
            if self.batch_by_clade:
                return self.guide_batch_by_clade(datasets, patristic_matrix, cladistic_matrix, data_blosum,
                                                 batch_blosum)
            else:

                return self.guide_batch(datasets, patristic_matrix, cladistic_matrix, data_blosum,
                                            batch_blosum=None,map_estimates=map_estimates)
        else:
            return self.guide_noplating(datasets, patristic_matrix, cladistic_matrix, data_blosum,
                                        batch_blosum=None,map_estimates=map_estimates)

    def guide_noplating(self,datasets, patristic_matrix_sorted,cladistic_matrix,data_blosum,batch_blosum=None,map_estimates=None):
        """
        :param tensor data_blosum here is the ENTIRE data encoded in blosum vector form instead of integers ---> EQUAL to self.dataset_train_blosum
        """
        #aminoacid_sequences = datasets["blosum"][:, 2:, 0]

        alpha = self.alpha
        sigma_n = self.sigma_n
        sigma_f = self.sigma_f
        lambd = self.lambd


        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            #Highlight: embed the amino acids represented by their respective blosum scores (data_blosim=self.dataset_train_blosum)
            aminoacid_sequences = self.embeddingencoder(self.dataset_train_blosum) #remember for the corals the aa_prob is 24 #TODO: Change to datasets["blosum"]

            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            encoder_output = self.encoder(aminoacid_sequences,encoder_h_0) #[n,z_dim]
            z_loc, z_scale = encoder_output["z_loc"], encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z",dist.Normal(z_loc.T,z_scale.T)).to_event(1) #[z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim,aminoacid_sequences.shape[0])


        return {"alpha":alpha,
                "sigma_n":sigma_n,
                "sigma_f":sigma_f,
                "lambd":lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "rnn_final_bidirectional": encoder_output["rnn_final_bidirectional"],
                "rnn_final_hidden_state": encoder_output["rnn_final_hidden_state"],
                "rnn_hidden_states": encoder_output["rnn_hidden_states"],
                }

    def guide_batch(self, datasets, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param tensor data_blosum here is the BATCH data encoded in blosum vector form instead of integers
        """
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        # aminoacid_sequences = datasets["blosum"][:, 2:, 0]

        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            # alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(1))
            # sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
            # sigma_n = pyro.sample("sigma_n",dist.HalfNormal(alpha[1]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # Gaussian noise
            # lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # characteristic length-scale
            #with pyro.poutine.scale(scale=map_estimates["annealing_factor"] if map_estimates is not None else 1):
            alpha = pyro.sample("alpha", dist.Delta(self.alpha).to_event(1))
            sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f).to_event(1))
            lambd = pyro.sample("lambd", dist.Delta(self.lambd).to_event(1))
            # Highlight: embed the amino acids represented by their respective blosum scores
            aminoacid_sequences = self.embeddingencoder(datasets["blosum"])  # remember for the corals the aa_prob is 24
            # aminoacid_sequences = self.dataset_train_blosum
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
            encoder_output = self.encoder(aminoacid_sequences, encoder_h_0)  # [n,z_dim]
            z_loc,z_scale = encoder_output["z_loc"],encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim, aminoacid_sequences.shape[0])

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "rnn_final_bidirectional":encoder_output["rnn_final_bidirectional"],
                "rnn_final_forward_backward_sum":encoder_output["rnn_final_forward_backward_sum"],
                "rnn_final_hidden_state": encoder_output["rnn_final_hidden_state"],
                "rnn_hidden_states": encoder_output["rnn_hidden_states"],
                }

    def guide_batch_experiment(self, datasets, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param tensor data_blosum here is the BATCH data encoded in blosum vector form instead of integers
        """
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        # aminoacid_sequences = datasets["blosum"][:, 2:, 0]

        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):

            alpha = pyro.sample("alpha", dist.Delta(self.alpha).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f).to_event(1))
            r = pyro.sample("r", dist.Delta(self.r).to_event(1))
            # Highlight: embed the amino acids represented by their respective blosum scores
            aminoacid_sequences = self.embeddingencoder(datasets["blosum"])  # remember for the corals the aa_prob is 24
            # aminoacid_sequences = self.dataset_train_blosum
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
            encoder_output = self.encoder(aminoacid_sequences, encoder_h_0)  # [n,z_dim]
            z_loc,z_scale = encoder_output["z_loc"],encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim, aminoacid_sequences.shape[0])

        return {"alpha": alpha,
                "sigma_f": sigma_f,
                "r": r,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "rnn_final_bidirectional":encoder_output["rnn_final_bidirectional"],
                "rnn_final_forward_backward_sum":encoder_output["rnn_final_forward_backward_sum"],
                "rnn_final_hidden_state": encoder_output["rnn_final_hidden_state"],
                "rnn_hidden_states": encoder_output["rnn_hidden_states"],
                }

    def guide_batch_by_clade(self, datasets, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):

        """
        :param tensor data_blosum here is the CLADE-BATCHED data in blosum vector form instead of integers
        :param batch_blosum is the weighted average of the blosum vectors for the clade per column site in the MSA"""
        # aminoacid_sequences = datasets[:, 2:, 0]
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
            sigma_n = pyro.sample("sigma_n",dist.HalfNormal(alpha[1]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # Gaussian noise
            lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # characteristic length-scale
            # Highlight: embed the amino acids represented by their respective blosum scores
            aminoacid_sequences = self.embeddingencoder(data_blosum)  # remember for the corals the aa_prob is 24
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],
                                                self.draupnir.gru_hidden_dim).contiguous()
            encoder_output = self.encoder(aminoacid_sequences, encoder_h_0)  # [n,z_dim]
            z_loc, z_scale = encoder_output["z_loc"], encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim, aminoacid_sequences.shape[0])

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "rnn_final_bidirectional": encoder_output["rnn_final_bidirectional"],
                "rnn_final_hidden_state": encoder_output["rnn_final_hidden_state"],
                "rnn_hidden_states": encoder_output["rnn_hidden_states"]
                }

class DRAUPNIRGuides_classic_1b(DRAUPNIRGUIDES):
    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        DRAUPNIRGUIDES.__init__(self,draupnir_model,ModelLoad, Draupnir)

        self.encoder = RNNEncoder(align_seq_len=self.draupnir.align_seq_len,
                                  aa_prob=self.draupnir.aa_probs,
                                  n_leaves=self.draupnir.n_leaves,
                                  gru_hidden_dim=self.draupnir.gru_hidden_dim,
                                  z_dim=self.draupnir.z_dim,
                                  input_size=self.encoder_input_size,
                                  kappa_addition=self.draupnir.kappa_addition,
                                  num_layers=self.draupnir.num_layers,
                                  pretrained_params=self.draupnir.pretrained_params)
        self.embeddingencoder = EmbedComplexEncoder(input_dim=self.draupnir.aa_probs,
                                                    embedding_dim=self.draupnir.embedding_dim,
                                                    out_dim=self.draupnir.aa_probs)

    def guide(self, datasets, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param patristic_matrix: matrix of patristic distances (branch lengths) between the nodes in the tree
        :param cladistic_matrix: matrix of cladistic distances between the nodes in the tree
        :param data_blosum : data encoded with blosum vectors
        :param batch_blosum : weighted average of blosum scores per column alignment for a batch of sequences"""


        if self.batch_size == None or self.batch_size > 1:
            if self.batch_by_clade:
                return self.guide_batch_by_clade(datasets, patristic_matrix, cladistic_matrix, data_blosum,
                                                 batch_blosum)
            else:

                return self.guide_batch(datasets, patristic_matrix, cladistic_matrix, data_blosum,
                                            batch_blosum=None,map_estimates=map_estimates)
        else:
            return self.guide_noplating(datasets, patristic_matrix, cladistic_matrix, data_blosum,
                                        batch_blosum=None,map_estimates=map_estimates)

    def guide_noplating(self,datasets, patristic_matrix_sorted,cladistic_matrix,data_blosum,batch_blosum=None,map_estimates=None):
        """
        :param tensor data_blosum here is the ENTIRE data encoded in blosum vector form instead of integers ---> EQUAL to self.dataset_train_blosum
        """
        #aminoacid_sequences = datasets["blosum"][:, 2:, 0]

        alpha = self.alpha
        sigma_n = self.sigma_n
        sigma_f = self.sigma_f
        lambd = self.lambd


        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            #Highlight: embed the amino acids represented by their respective blosum scores (data_blosim=self.dataset_train_blosum)
            aminoacid_sequences = self.embeddingencoder(self.dataset_train_blosum) #remember for the corals the aa_prob is 24 #TODO: Change to datasets["blosum"]

            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            encoder_output = self.encoder(aminoacid_sequences,encoder_h_0) #[n,z_dim]
            z_loc, z_scale = encoder_output["z_loc"], encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z",dist.Normal(z_loc.T,z_scale.T)).to_event(1) #[z_dim,n]

            assert latent_z.shape == (self.draupnir.z_dim,aminoacid_sequences.shape[0])


        return {"alpha":alpha,
                "sigma_n":sigma_n,
                "sigma_f":sigma_f,
                "lambd":lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "rnn_final_bidirectional": encoder_output["rnn_final_bidirectional"],
                "rnn_final_hidden_state": encoder_output["rnn_final_hidden_state"],
                "rnn_hidden_states": encoder_output["rnn_hidden_states"],
                }

    def guide_batch(self, datasets, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param tensor data_blosum here is the BATCH data encoded in blosum vector form instead of integers
        """
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        # aminoacid_sequences = datasets["blosum"][:, 2:, 0]

        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            # alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(1))
            # sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
            # sigma_n = pyro.sample("sigma_n",dist.HalfNormal(alpha[1]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # Gaussian noise
            # lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # characteristic length-scale
            #with pyro.poutine.scale(scale=map_estimates["annealing_factor"] if map_estimates is not None else 1):
            alpha = pyro.sample("alpha", dist.Delta(self.alpha).to_event(1))
            sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f).to_event(1))
            lambd = pyro.sample("lambd", dist.Delta(self.lambd).to_event(1))
            # Highlight: embed the amino acids represented by their respective blosum scores
            aminoacid_sequences = self.embeddingencoder(datasets["blosum"])  # remember for the corals the aa_prob is 24
            # aminoacid_sequences = self.dataset_train_blosum
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
            encoder_output = self.encoder(aminoacid_sequences, encoder_h_0)  # [n,z_dim]
            z_loc,z_scale = encoder_output["z_loc"],encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim, aminoacid_sequences.shape[0])

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "rnn_final_bidirectional":encoder_output["rnn_final_bidirectional"],
                "rnn_final_forward_backward_sum":encoder_output["rnn_final_forward_backward_sum"],
                "rnn_final_hidden_state": encoder_output["rnn_final_hidden_state"],
                "rnn_hidden_states": encoder_output["rnn_hidden_states"],
                }

    def guide_batch_by_clade(self, datasets, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):

        """
        :param tensor data_blosum here is the CLADE-BATCHED data in blosum vector form instead of integers
        :param batch_blosum is the weighted average of the blosum vectors for the clade per column site in the MSA"""
        # aminoacid_sequences = datasets[:, 2:, 0]
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
            sigma_n = pyro.sample("sigma_n",dist.HalfNormal(alpha[1]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # Gaussian noise
            lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # characteristic length-scale
            # Highlight: embed the amino acids represented by their respective blosum scores
            aminoacid_sequences = self.embeddingencoder(data_blosum)  # remember for the corals the aa_prob is 24
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],
                                                self.draupnir.gru_hidden_dim).contiguous()
            encoder_output = self.encoder(aminoacid_sequences, encoder_h_0)  # [n,z_dim]
            z_loc, z_scale = encoder_output["z_loc"], encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim, aminoacid_sequences.shape[0])

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "rnn_final_bidirectional": encoder_output["rnn_final_bidirectional"],
                "rnn_final_hidden_state": encoder_output["rnn_final_hidden_state"],
                "rnn_hidden_states": encoder_output["rnn_hidden_states"]
                }

class DRAUPNIRGuides_transformer(DRAUPNIRGUIDES):
    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        DRAUPNIRGUIDES.__init__(self,draupnir_model,ModelLoad, Draupnir)

        adapted_input_dim = self.draupnir.aa_probs if self.draupnir.aa_probs % 2 == 0 else self.draupnir.aa_probs + 1  # this is necessary for MHA
        self.encoder = TransformerEncoder3(input_dim=self.draupnir.aa_probs,
                                           adapted_input_dim=adapted_input_dim,
                                           align_seq_len=self.draupnir.align_seq_len,
                                           output_dim=self.draupnir.z_dim)
        self.embeddingencoder = EmbedComplexEncoder(self.draupnir.aa_probs, self.draupnir.embedding_dim,
                                                    adapted_input_dim)
        self.positional_encodings = PositionalEncodings(self.draupnir.align_seq_len, self.draupnir.aa_probs,
                                                        adapted_input_dim, 10000) #todo: i think the positional encodings i put them inside

    def guide(self, datasets, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param tensor data_blosum here is the BATCH data encoded in blosum vector form instead of integers
        """
        #pyro.module("encoder", self.encoder) #the transformer is only the scaled dot attention
        pyro.module("embeddingsencoder", self.embeddingencoder)
        pyro.module("encoder", self.encoder)
        pyro.module("positional_encodings", self.positional_encodings)
        # aminoacid_sequences = datasets["blosum"][:, 2:, 0]
        aa_sequences = datasets["blosum"]
        nseqs = aa_sequences.shape[0]
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            # alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(1))
            # sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
            # sigma_n = pyro.sample("sigma_n",dist.HalfNormal(alpha[1]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # Gaussian noise
            # lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.draupnir.z_dim, ]).to_event(1))  # characteristic length-scale

            alpha = pyro.sample("alpha", dist.Delta(self.alpha).to_event(1))
            sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f).to_event(1))
            lambd = pyro.sample("lambd", dist.Delta(self.lambd).to_event(1))
            # Highlight: embed the amino acids represented by their respective blosum scores
            aminoacid_sequences = self.embeddingencoder(aa_sequences)  # Highlight: i) Makes linear projection ii) Makes the feature dimensions even to be able to apply the rotational embeddings
            #queryw, keyw = self.positional_encodings.apply(aminoacid_sequences)
            # sinusoidal_encodings = self.positional_encodings.sinusoidal_encodings() #[1, L, feat_dim]
            # aminoacid_sequences = aminoacid_sequences+ sinusoidal_encodings #sum because independent set of vectors with high likelihood

            #encoder_output = self.encoder.forward(queryw,keyw,aminoacid_sequences,None)  # [n,z_dim] #TODO: Introduce masking

            #todo: there is no start token here
            encoder_output = self.encoder.forward(aminoacid_sequences,None)  # [n,z_dim] #TODO: Introduce masking
            z_loc,z_scale = encoder_output["z_loc"],encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]


            assert latent_z.shape == (self.draupnir.z_dim, nseqs), f"expected shape ({self.draupnir.z_dim}, {nseqs}), found {latent_z.shape}"

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "hidden_states": encoder_output["hidden_states"].detach(),
                # "attention_scores" : encoder_output["attention_scores"].detach(),
                "context_vector": encoder_output["context_vector"].detach(),
                # "attention_logits": encoder_output["attention_logits"].detach()
                }

class DRAUPNIRGuides_z_esm(DRAUPNIRGUIDES):
    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        DRAUPNIRGUIDES.__init__(self,draupnir_model,ModelLoad, Draupnir)

        self.encoder = FCLEncoder(align_seq_len=self.draupnir.align_seq_len,
                                  aa_prob=self.draupnir.aa_probs,
                                  n_leaves=self.draupnir.n_leaves,
                                  gru_hidden_dim=self.draupnir.gru_hidden_dim,
                                  z_dim=self.draupnir.z_dim,
                                  input_size=self.draupnir.embedding_dim,
                                  num_layers=self.draupnir.num_layers)

    def guide(self, datasets, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param patristic_matrix: matrix of patristic distances (branch lengths) between the nodes in the tree
        :param cladistic_matrix: matrix of cladistic distances between the nodes in the tree
        :param data_blosum : data encoded with blosum vectors
        :param batch_blosum : weighted average of blosum scores per column alignment for a batch of sequences"""

        pyro.module("encoder", self.encoder)
        # aminoacid_sequences = datasets["int"][:, 2:, 0]
        aa_sequences = datasets["blosum"]  # blosum does not contain the indexes
        nseqs = aa_sequences.shape[0]
        esm_embeddings = datasets["embedding"][:, 2:]  # the indexes are in [:,0,1]
        esm_representations = datasets["sequences_representations"][:, 1:]  # the indexes are in [:,0]

        # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            alpha = pyro.sample("alpha", dist.Delta(self.alpha).to_event(1))
            sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f).to_event(1))
            lambd = pyro.sample("lambd", dist.Delta(self.lambd).to_event(1))
            # Highlight: embed the amino acids represented by their respective blosum scores

            # aminoacid_embeddings = self.embeddingencoder(esm_embeddings) #i use the "aligned embeddings"
            # encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_embeddings.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            # encoder_output = self.encoder(aminoacid_embeddings, encoder_h_0)  # [n,z_dim] #todo: i need the seq lens if i use unaligned sequences

            encoder_output = self.encoder(esm_representations, esm_representations)  # [n,z_dim]

            z_loc, z_scale = encoder_output["z_loc"], encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]

            assert latent_z.shape == (self.draupnir.z_dim,
                                      nseqs), f"expected shape ({self.draupnir.z_dim}, {nseqs}), found {latent_z.shape}"

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                }

class DRAUPNIRGuides_hidden_esm(DRAUPNIRGUIDES):
    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        DRAUPNIRGUIDES.__init__(self,draupnir_model,ModelLoad, Draupnir)

        self.encoder = RNNEncoder(align_seq_len=self.draupnir.align_seq_len,
                                  aa_prob=self.draupnir.aa_probs,
                                  n_leaves=self.draupnir.n_leaves,
                                  gru_hidden_dim=self.draupnir.gru_hidden_dim,
                                  z_dim=self.draupnir.z_dim,
                                  input_size=self.draupnir.gru_hidden_dim,
                                  kappa_addition=self.draupnir.kappa_addition,
                                  num_layers=self.draupnir.num_layers,
                                  pretrained_params=self.draupnir.pretrained_params)
        self.embeddingencoder = EmbedComplexEncoder(input_dim=self.draupnir.embedding_dim,
                                                    embedding_dim=self.draupnir.gru_hidden_dim,
                                                    out_dim=self.draupnir.gru_hidden_dim)

    def guide(self, datasets, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param patristic_matrix: matrix of patristic distances (branch lengths) between the nodes in the tree
        :param cladistic_matrix: matrix of cladistic distances between the nodes in the tree
        :param data_blosum : data encoded with blosum vectors
        :param batch_blosum : weighted average of blosum scores per column alignment for a batch of sequences"""

        pyro.module("encoder", self.encoder)
        # aminoacid_sequences = datasets["int"][:, 2:, 0]
        aa_sequences = datasets["blosum"]#blosum does not contain the indexes
        nseqs = aa_sequences.shape[0]
        esm_embeddings = datasets["embedding"][:,2:] #the indexes are in [:,0,1]
        esm_representations = datasets["sequences_representations"][:,1:] #the indexes are in [:,0]


        # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):

            alpha = pyro.sample("alpha", dist.Delta(self.alpha).to_event(1))
            sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f).to_event(1))
            lambd = pyro.sample("lambd", dist.Delta(self.lambd).to_event(1))
            # Highlight: embed the amino acids represented by their respective blosum scores

            aminoacid_embeddings = self.embeddingencoder(esm_embeddings) #i use the "aligned embeddings"
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_embeddings.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            encoder_output = self.encoder(aminoacid_embeddings, encoder_h_0)  # [n,z_dim] #todo: i need the seq lens if i use unaligned sequences

            #encoder_output = self.encoder(esm_representations,esm_representations) # [n,z_dim]

            z_loc,z_scale = encoder_output["z_loc"],encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]


            assert latent_z.shape == (self.draupnir.z_dim, nseqs), f"expected shape ({self.draupnir.z_dim}, {nseqs}), found {latent_z.shape}"

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                }

class DRAUPNIRGuides_xlstm(DRAUPNIRGUIDES):
    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        DRAUPNIRGUIDES.__init__(self,draupnir_model,ModelLoad, Draupnir)

        self.encoder1 = xLSTMEncoder(max_len=self.draupnir.align_seq_len,
                                  input_size=self.draupnir.z_dim,
                                    z_dim = self.draupnir.z_dim)

        self.encoder2 = RNNEncoder(align_seq_len=self.draupnir.align_seq_len,
                                  aa_prob=self.draupnir.aa_probs,
                                  n_leaves=self.draupnir.n_leaves,
                                  gru_hidden_dim=self.draupnir.gru_hidden_dim,
                                  z_dim=self.draupnir.z_dim,
                                  input_size=self.draupnir.z_dim, #self.encoder_input_size,
                                  kappa_addition=self.draupnir.kappa_addition,
                                  num_layers=self.draupnir.num_layers,
                                  pretrained_params=self.draupnir.pretrained_params)

        self.embeddingencoder = EmbedComplexEncoder(input_dim=self.draupnir.aa_probs,
                                                    embedding_dim=self.draupnir.gru_hidden_dim,
                                                    out_dim=self.draupnir.z_dim)

    def guide(self, datasets, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param patristic_matrix: matrix of patristic distances (branch lengths) between the nodes in the tree
        :param cladistic_matrix: matrix of cladistic distances between the nodes in the tree
        :param data_blosum : data encoded with blosum vectors
        :param batch_blosum : weighted average of blosum scores per column alignment for a batch of sequences"""

        pyro.module("encoder1", self.encoder1)
        pyro.module("encoder2", self.encoder2)
        pyro.module("embeddingencoder", self.embeddingencoder)
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
        aa_sequences_blosum = datasets["blosum"]#blosum does not contain the indexes
        nseqs = aa_sequences_blosum.shape[0]

        # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):

            alpha = pyro.sample("alpha", dist.Delta(self.alpha).to_event(1))
            sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f).to_event(1))
            lambd = pyro.sample("lambd", dist.Delta(self.lambd).to_event(1))
            # Highlight: embed the amino acids represented by their respective blosum scores

            aminoacid_embeddings_0 = self.embeddingencoder(aa_sequences_blosum)

            encoder_1_output = self.encoder1(aminoacid_embeddings_0)
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder2.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()

            #todo:
            aminoacid_embeddings_0 = aminoacid_embeddings_0 + encoder_1_output["embeddings"]
            encoder_2_output = self.encoder2(aminoacid_embeddings_0,encoder_h_0)  # [n,z_dim] #todo: i need the seq lens if i use unaligned sequences


            z_loc,z_scale = encoder_2_output["z_loc"],encoder_2_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]


            assert latent_z.shape == (self.draupnir.z_dim, nseqs), f"expected shape ({self.draupnir.z_dim}, {nseqs}), found {latent_z.shape}"

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "embeddings": encoder_1_output["embeddings"],
                "batch_nodes" :batch_nodes
                }

class DRAUPNIRGuides_minrnn(DRAUPNIRGUIDES):
    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        DRAUPNIRGUIDES.__init__(self,draupnir_model,ModelLoad, Draupnir)


        self.embeddingencoder = EmbedComplexEncoder(input_dim=self.draupnir.aa_probs,
                                                    embedding_dim=self.draupnir.gru_hidden_dim,
                                                    out_dim=self.draupnir.z_dim) #todo: can be bigger
        self.encoder = miniGRUEncoder(depth=2,
                                      input_dim = self.draupnir.z_dim,
                                      output_dim = self.draupnir.z_dim)


    def guide(self, datasets, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param patristic_matrix: matrix of patristic distances (branch lengths) between the nodes in the tree
        :param cladistic_matrix: matrix of cladistic distances between the nodes in the tree
        :param data_blosum : data encoded with blosum vectors
        :param batch_blosum : weighted average of blosum scores per column alignment for a batch of sequences"""

        pyro.module("encoder", self.encoder)
        pyro.module("embeddingencoder", self.embeddingencoder)
        aa_sequences_int = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
        aa_sequences_blosum = datasets["blosum"]#blosum does not contain the indexes
        nseqs = aa_sequences_blosum.shape[0]

        # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):

            alpha = pyro.sample("alpha", dist.Delta(self.alpha).to_event(1))
            sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n).to_event(1))
            sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f).to_event(1))
            lambd = pyro.sample("lambd", dist.Delta(self.lambd).to_event(1))
            # Highlight: embed the amino acids represented by their respective blosum scores

            aminoacid_embeddings = self.embeddingencoder(aa_sequences_blosum)
            prev_embeddings = map_estimates["embeddings"] if map_estimates is not None else None
            encoder_output = self.encoder(aminoacid_embeddings)  # [n,z_dim] #todo: i need the seq lens if i use unaligned sequences

            z_loc,z_scale = encoder_output["z_loc"],encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z", dist.Normal(z_loc.T, z_scale.T))  # [z_dim,n]


            assert latent_z.shape == (self.draupnir.z_dim, nseqs), f"expected shape ({self.draupnir.z_dim}, {nseqs}), found {latent_z.shape}"

        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "z_loc": z_loc,
                "z_scale": z_scale,
                "latent_z": latent_z,
                "embeddings": encoder_output["embeddings"],
                "batch_nodes" :batch_nodes
                }































