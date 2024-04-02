"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
from types import MethodType

import pyro
from pyro.contrib.easyguide import EasyGuide
from pyro.nn import PyroParam
import torch.nn as nn
import torch
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from draupnir.models import *
import draupnir.models_utils as DraupnirModelsUtils
class DRAUPNIRGUIDES(EasyGuide):
    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        super(DRAUPNIRGUIDES, self).__init__(draupnir_model)
        self.guide_type = ModelLoad.args.select_guide
        self.draupnir = Draupnir
        self.encoder_rnn_input_size = self.draupnir.aa_probs
        self.dataset_train_blosum = self.draupnir.dataset_train_blosum
        self.batch_size = self.draupnir.batch_size
        self.batch_by_clade = self.draupnir.batch_by_clade
        if self.draupnir.pretrained_params is not None:
            self.h_0_GUIDE = nn.Parameter(self.draupnir.pretrained_params["h_0_GUIDE"], requires_grad=True).to(self.draupnir.device)
        else:
            self.h_0_GUIDE = nn.Parameter(torch.randn(self.draupnir.gru_hidden_dim), requires_grad=True).to(self.draupnir.device)
        self.encoder = RNNEncoder(self.draupnir.align_seq_len, self.draupnir.aa_probs,self.draupnir.n_leaves, self.draupnir.gru_hidden_dim, self.draupnir.z_dim, self.encoder_rnn_input_size,self.draupnir.kappa_addition,self.draupnir.num_layers,self.draupnir.pretrained_params)
        self.embeddingencoder = EmbedComplexEncoder(self.draupnir.aa_probs,self.draupnir.embedding_dim,self.draupnir.pretrained_params)
        self.alpha = PyroParam(dist.HalfNormal(torch.tensor([1.0])).sample([3]),constraint=constraints.positive,event_dim=0) #constraint=constraints.interval(0., 10.)--->TODO:Event dimension??
        self.sigma_n = PyroParam(dist.HalfNormal(torch.tensor([1.0])).sample([self.draupnir.z_dim]),constraint=constraints.positive,event_dim=0)
        self.sigma_f = PyroParam(dist.HalfNormal(torch.tensor([1.0])).sample([self.draupnir.z_dim]),constraint=constraints.positive,event_dim=0)
        self.lambd = PyroParam(dist.HalfNormal(torch.tensor([1.0])).sample([self.draupnir.z_dim]),constraint=constraints.positive,event_dim=0)

        if self.draupnir.plating:
            self.encoder_splitted_leaves_indexes = list(torch.tensor_split(torch.arange(self.draupnir.n_leaves), int(self.draupnir.n_leaves / self.draupnir.plate_size)) * self.draupnir.num_epochs)

    def guide(self, family_data, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param patristic_matrix: matrix of patristic distances (branch lengths) between the nodes in the tree
        :param cladistic_matrix: matrix of cladistic distances between the nodes in the tree
        :param data_blosum : data encoded with blosum vectors
        :param batch_blosum : weighted average of blosum scores per column alignment for a batch of sequences"""
        if self.batch_size == None or self.batch_size > 1:
            if self.batch_by_clade:
                return self.guide_batch_by_clade(family_data, patristic_matrix, cladistic_matrix, data_blosum,
                                                 batch_blosum)
            else:
                return self.guide_batch(family_data, patristic_matrix, cladistic_matrix, data_blosum,
                                        batch_blosum=None)
        else:
            return self.guide_noplating(family_data, patristic_matrix, cladistic_matrix, data_blosum,
                                        batch_blosum=None)

    def guide_noplating(self,family_data, patristic_matrix_sorted,cladistic_matrix,data_blosum,batch_blosum=None,map_estimates=None):
        """
        :param tensor data_blosum here is the ENTIRE data encoded in blosum vector form instead of integers ---> EQUAL to self.dataset_train_blosum
        """
        #aminoacid_sequences = family_data[:, 2:, 0]

        alpha = self.alpha
        sigma_n = self.sigma_n
        sigma_f = self.sigma_f
        lambd = self.lambd


        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        with pyro.plate("plate_batch", dim=-1, device=self.draupnir.device):
            #Highlight: embed the amino acids represented by their respective blosum scores (data_blosim=self.dataset_train_blosum)
            aminoacid_sequences = self.embeddingencoder(self.dataset_train_blosum) #remember for the corals the aa_prob is 24
            #aminoacid_sequences = self.dataset_train_blosum
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            encoder_output = self.encoder(aminoacid_sequences,encoder_h_0) #[n,z_dim]
            z_loc, z_scale = encoder_output["z_loc"], encoder_output["z_scale"]
            latent_z = pyro.sample("latent_z",dist.Normal(z_loc.T,z_scale.T)).to_event(1) #[z_dim,n]
            print("Guide Latent space: {}".format(latent_z.shape))
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

    def guide_batch(self, family_data, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):
        """
        :param tensor data_blosum here is the BATCH data encoded in blosum vector form instead of integers
        """
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        # aminoacid_sequences = family_data[:, 2:, 0]
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
            aminoacid_sequences = self.embeddingencoder(data_blosum)  # remember for the corals the aa_prob is 24
            # aminoacid_sequences = self.dataset_train_blosum
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],
                                                self.draupnir.gru_hidden_dim).contiguous()
            # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
            #with pyro.plate("plate_guide", aminoacid_sequences.shape[0], dim=-1):
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
                "rnn_final_hidden_state": encoder_output["rnn_final_hidden_state"],
                "rnn_hidden_states": encoder_output["rnn_hidden_states"],
                }

    def guide_batch_by_clade(self, family_data, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum=None,map_estimates=None):

        """
        :param tensor data_blosum here is the CLADE-BATCHED data in blosum vector form instead of integers
        :param batch_blosum is the weighted average of the blosum vectors for the clade per column site in the MSA"""
        # aminoacid_sequences = family_data[:, 2:, 0]
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




