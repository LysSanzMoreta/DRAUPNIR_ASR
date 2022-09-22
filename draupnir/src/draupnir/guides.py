"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
from pyro.contrib.easyguide import EasyGuide
from pyro.nn import PyroParam
from draupnir.models import *
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
        self.alpha = PyroParam(torch.randn(3),constraint=constraints.positive,event_dim=1) #constraint=constraints.interval(0., 10.)--->TODO:Event dimension??
        self.sigma_n = PyroParam(torch.randn(self.draupnir.z_dim),constraint=constraints.positive, event_dim=1) #TODO: Change to randn
        self.sigma_f = PyroParam(torch.randn(self.draupnir.z_dim),constraint=constraints.positive,event_dim=1)
        self.lambd = PyroParam(torch.randn(self.draupnir.z_dim),constraint=constraints.positive, event_dim=1)
        if self.draupnir.plating:
            self.encoder_splitted_leaves_indexes = list(torch.tensor_split(torch.arange(self.draupnir.n_leaves), int(self.draupnir.n_leaves / self.draupnir.plate_size)) * self.draupnir.num_epochs)

    def guide(self, family_data, patristic_matrix, cladistic_matrix, data_blosum, batch_blosum):
        """data_blosum is the data encoded with blosum vectors
        batch_blosum is the weighted average of blosum scores per column alignment for a batch of sequences"""
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

    def guide_noplating(self,family_data, patristic_matrix_sorted,cladistic_matrix,data_blosum,batch_blosum):
        """
        :param tensor data_blosum here is the ENTIRE data encoded in blosum vector form instead of integers ---> EQUAL to self.dataset_train_blosum
        """
        #aminoacid_sequences = family_data[:, 2:, 0]
        #Highlight: Initialize variational parameters of the OU process
        alpha = pyro.sample("alpha", dist.Delta(self.alpha))
        sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n))
        sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f))
        lambd = pyro.sample("lambd", dist.Delta(self.lambd))
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        #Highlight: embed the amino acids represented by their respective blosum scores (data_blosim=self.dataset_train_blosum)
        aminoacid_sequences = self.embeddingencoder(self.dataset_train_blosum) #remember for the corals the aa_prob is 24
        #aminoacid_sequences = self.dataset_train_blosum
        encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()
        #Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
        with pyro.plate("plate_guide", aminoacid_sequences.shape[0], dim=-1):
            z_mean,z_variance = self.encoder(aminoacid_sequences,encoder_h_0) #[n,z_dim]
            latent_z = pyro.sample("latent_z",dist.Normal(z_mean.T,z_variance.T)) #[z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim,aminoacid_sequences.shape[0])
        return {"alpha":alpha,
                "sigma_n":sigma_n,
                "sigma_f":sigma_f,
                "lambd":lambd,
                "latent_z":latent_z}

    def guide_batch(self, family_data, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum):
        """
        :param tensor data_blosum here is the BATCH data encoded in blosum vector form instead of integers
        """
        # aminoacid_sequences = family_data[:, 2:, 0]
        # Highlight: Initialize variational parameters of the OU process
        alpha = pyro.sample("alpha", dist.Delta(self.alpha))
        sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n))
        sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f))
        lambd = pyro.sample("lambd", dist.Delta(self.lambd))
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        # Highlight: embed the amino acids represented by their respective blosum scores
        aminoacid_sequences = self.embeddingencoder(data_blosum)  # remember for the corals the aa_prob is 24
        # aminoacid_sequences = self.dataset_train_blosum
        encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],
                                            self.draupnir.gru_hidden_dim).contiguous()
        # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
        with pyro.plate("plate_guide", aminoacid_sequences.shape[0], dim=-1):
            z_mean, z_variance = self.encoder(aminoacid_sequences, encoder_h_0)  # [n,z_dim]
            latent_z = pyro.sample("latent_z", dist.Normal(z_mean.T, z_variance.T))  # [z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim, aminoacid_sequences.shape[0])
        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "latent_z": latent_z}

    def guide_batch_by_clade(self, family_data, patristic_matrix_sorted, cladistic_matrix, data_blosum, batch_blosum):

        """
        :param tensor data_blosum here is the CLADE-BATCHED data in blosum vector form instead of integers
        :param batch_blosum is the weighted average of the blosum vectors for the clade per column site in the MSA"""
        # aminoacid_sequences = family_data[:, 2:, 0]
        # Highlight: Initialize variational parameters of the OU process
        alpha = pyro.sample("alpha", dist.Delta(self.alpha))
        sigma_n = pyro.sample("sigma_n", dist.Delta(self.sigma_n))
        sigma_f = pyro.sample("sigma_f", dist.Delta(self.sigma_f))
        lambd = pyro.sample("lambd", dist.Delta(self.lambd))
        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        # Highlight: embed the amino acids represented by their respective blosum scores
        aminoacid_sequences = self.embeddingencoder(data_blosum)  # remember for the corals the aa_prob is 24
        encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],
                                            self.draupnir.gru_hidden_dim).contiguous()
        # Highlight: Everything, n_leaves and n_z, is independent (we can plate over any of them , is fine)
        with pyro.plate("plate_guide", aminoacid_sequences.shape[0], dim=-1):
            z_mean, z_variance = self.encoder(aminoacid_sequences, encoder_h_0)  # [n,z_dim]
            latent_z = pyro.sample("latent_z", dist.Normal(z_mean.T, z_variance.T))  # [z_dim,n]
            assert latent_z.shape == (self.draupnir.z_dim, aminoacid_sequences.shape[0])
        return {"alpha": alpha,
                "sigma_n": sigma_n,
                "sigma_f": sigma_f,
                "lambd": lambd,
                "latent_z": latent_z}




