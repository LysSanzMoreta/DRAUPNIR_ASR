import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import pyro
from pyro import distributions as dist
from operator import itemgetter
from collections import namedtuple
from pyro.contrib.easyguide import EasyGuide
from pyro.nn import PyroParam

SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi","covariance"])

class GPKernel(ABC):
    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OUKernel_Fast(GPKernel):
    """ Kernel that computes the covariance matrix for a z Ornstein Ulenbeck processes. As stated in Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
    :param tensor sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
    :param tensor lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes),larger l implies that the noise should be bigger to capture big point fluctuations
    :param tensor sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
    **References:**
    "Ancestral Inference from Functional Data: Statistical Methods and Numerical Examples"
    """

    def __init__(self, sigma_f, lambd, sigma_n=None, z_dim=30, kernel_type="3"):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lambd = lambd
        self.kernel_type = kernel_type
        self.z_dim = z_dim


    def prior0(self, t: torch.Tensor) -> torch.Tensor:

        assert self.sigma_f is None, "sigma_f should be None"
        assert self.lambd is None, "sigma_f should be None"
        assert self.sigma_n is None, "sigma_n should be None"

        t = t.repeat(self.z_dim, 1, 1)

        return torch.exp(-t)

    def prior1(self, t: torch.Tensor) -> torch.Tensor:  # original prior

        assert self.sigma_f is not None, "sigma_f cannot be None"
        assert self.sigma_n is not None, "sigma_n cannot be None"
        assert self.lambd is not None, "lambda cannot be None"

        lambd = self.lambd.unsqueeze(-1).unsqueeze(-1)  # self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1)  # [:,None,None]
        noise = torch.eye(t.shape[0])  # distributes noise/stochascity to diagonal of the covariance
        sigma_n = self.sigma_n.unsqueeze(-1).unsqueeze(-1)
        return first_term * second_term + sigma_n ** 2 * noise

    def prior2(self, t: torch.Tensor) -> torch.Tensor:

        assert self.sigma_f is not None, "sigma_f cannot be None"
        assert self.lambd is not None, "lambda cannot be None"

        lambd = self.lambd.unsqueeze(-1).unsqueeze(-1)  # self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1)  # [:,None,None]
        noise = torch.eye(t.shape[0])  # distributes noise/stochascity to diagonal of the covariance

        return first_term * second_term + noise * 1e-6

    def prior3(self, t: torch.Tensor) -> torch.Tensor:

        assert self.lambd is not None, "lambda cannot be None"
        assert self.sigma_f is None, "sigma_f has been removed, should be None"

        lambd = self.lambd.unsqueeze(-1)  # .unsqueeze(-1) #self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)


        return second_term

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "0":
            return self.prior0(t)
        elif self.kernel_type == "1":
            return self.prior1(t)
        elif self.kernel_type == "2":
            return self.prior2(t)
        elif self.kernel_type == "3":
            return self.prior3(t)
        else:
            raise ValueError(f"{self.kernel_type} not found")

class DRAUPNIRModelClass(nn.Module):

    def __init__(self,  ModelLoad):
        """
        :param namedtuple ModelLoad: named tuple with variables to load into the model
        """

        self.args = ModelLoad.args
        self.leaves_nodes = ModelLoad.leaves_nodes
        self.internal_nodes = ModelLoad.internal_nodes
        self.n_leaves = len(self.leaves_nodes)
        self.n_internal = len(self.internal_nodes)
        self.n_all = self.n_internal + self.n_leaves
        self.align_seq_len = ModelLoad.align_seq_len
        self.decoder = None #fill-in
        self.embed = None #fill-in


    def gp_prior(self,patristic_matrix): #unfinished example with the basic stuff
        """Whitening prior"""

        # apply a kernel function to the patristic matrix

        lambd = torch.ones((1,)) #dummy current prior example

        OU_covariance = OUKernel_Fast(None,lambd,None,kernel_type="3").forward(patristic_matrix)
        #do whitening or whatever ...

        latent_space = torch.zeros(self.n_leaves,self.z_dim) #dummy


        assert latent_space.shape == (self.n_leaves,self.z_dim)
        assert OU_covariance.shape == (self.n_leaves,self.n_leaves) #the current prior with whitening has this shape, the previous one was (self.z_dim,self.n_leaves,self.n_leaves)


        return {"latent_space": latent_space, "covariance": OU_covariance}

    def categorical_likelihood(self,likelihood_input): #i put the categorical as an example
        """Categorical likelihood
        :param dict likelihood_input
        """
        latent_space, decoder_hidden, aminoacid_sequences = itemgetter("latent_space", "decoder_hidden","aminoacid_sequences")(likelihood_input)
        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            logits = self.decoder.forward(
                input=latent_space,
                hidden=decoder_hidden)
            pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]


    def potts_likelihood(self,likelihood_input):
        """
        Low rank potts psudo likelihood
        """


    def likelihood(self):
        if self.args.likelihood_type=="categorical":
            return self.categorical_likelihood
        elif  self.args.likelihood_type=="potts":
            return self.potts_likelihood


    def model(self, datasets, patristic_matrix_train_sorted):
        """

        :param torch.tensor patristic_matrix_train_sorted: patristic matrix of shape (n_leaves + 1, n_leaves + 1), where the columns and the rows are the integer-encoded positions of the sequence in the tree (in tree level order). It is sorted in tree level order
        :param dict datasets: dictionary containing the integer-encoded sequences, the blosum encoded sequences and a few other things
        """


        aminoacid_sequences = datasets["int"][:, 2:, 0]
        # Highlight: Register NN modules
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-2, device=self.device):

            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior(patristic_matrix_train_sorted) # EXCHANGEABLE MODULE
            latent_space = out_dict["latent_space"]
            assert latent_space.shape == (self.n_leaves,self.z_dim)
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim] #This can maybe be done with new axis solely
            assert latent_space.shape == (self.n_leaves, self.align_seq_len ,self.z_dim)
            #this is the average blosum stuff, which is optional
            # blosum = self.blosum_weighted.repeat(latent_space.shape[0], 1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
            # blosum = self.embed(blosum)
            # latent_space = torch.cat((latent_space, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
            likelihood_input = dict(latent_space=latent_space,
                                    decoder_hidden=decoder_hidden,
                                    aminoacid_sequences=aminoacid_sequences)

            self.likelihood(likelihood_input) #EXCHANGEABLE MODULE

    def prediction_preprocessing(self, map_estimates, patristic_matrix_full, patristic_matrix_eval, batch_idx,use_test, use_test2):
        """Helper tool for the sampling of the internal/test or the leaves/train nodes"""

        if use_test or use_test2:  # Marginal posterior
            assert patristic_matrix_full[1:, 1:].shape == (self.n_all, self.n_all)
            out_prediction_dict = self.conditional_sampling(map_estimates, patristic_matrix_full)
            latent_space, covariance, internal_idx = out_prediction_dict["latent_space"], out_prediction_dict["covariance"], out_prediction_dict["internal_idx"]
            covariance = covariance[internal_idx]
            covariance = covariance[:, internal_idx]
            assert covariance.shape == (self.n_internal,self.n_internal)
        else:

            idx_train = (patristic_matrix_full[:, 0][..., None] == self.leaves_nodes).any(-1)
            idx_train[0] = True
            patristic_matrix_train = patristic_matrix_full[idx_train]
            patristic_matrix_train = patristic_matrix_train[:, idx_train]
            patristic_matrix_train = patristic_matrix_train[1:, 1:]  # remember to remove the node names
            lambd = torch.exp(map_estimates["log_lambd"])
            covariance = OUKernel_Fast(None, lambd, None, kernel_type="3").forward(patristic_matrix_train)
            L = torch.linalg.cholesky(covariance)
            assert covariance.shape == (self.n_leaves,self.n_leaves)

            latent_space = torch.matmul(L, map_estimates["eps_z"].T[:,:,None]).squeeze(-1).T
            assert latent_space.shape == (self.n_leaves, self.z_dim)


        return {"latent_space": latent_space, "covariance": covariance}



    def sampling_categorical_likelihood(self,sampling_input,use_argmax):


        latent_space_3d,decoder_hidden,n_samples = itemgetter("latent_space","decoder_hidden","n_samples")(sampling_input)
        logits = self.decoder.forward(
            input=latent_space_3d,
            hidden=decoder_hidden)
        if use_argmax:
            #Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
            aa_sequences = torch.argmax(logits,dim=2).unsqueeze(0) #I add one dimension at the beginning to resemble 1 sample and not have to change all the plotting code
            assert aa_sequences.shape == (self.n_internal,self.max_len)

        else:
            aa_sequences = dist.Categorical(logits=logits).sample([n_samples])
            assert  aa_sequences.shape == (n_samples,self.n_internal,self.max_len)

        return dict(logits=logits,
                    aa_sequences=aa_sequences)

    def sample(self, map_estimates, n_samples, patristic_matrix_full,patristic_matrix_eval,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """
        :param dict map_estimates: contains the guide output
        :param int n_samples contains the number of samples
        :param torch.tensor patristic_matrix_full with shape (n_train + n_test +1, n_train + n_test + 1)
        :param torch.tensor patristic_matrix_eval depending on the sampling it can be the train patristic matrix or the test
        """


        latent_space, covariance, n_nodes = itemgetter("latent_space", "covariance","n_nodes")(self.prediction_preprocessing(map_estimates, patristic_matrix_full, patristic_matrix_eval, batch_idx, use_test, use_test2)) #EXCHANGEABLE MODULE

        assert latent_space.shape == (self.n_internal,self.z_dim)

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_3d = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        # blosum = self.blosum_weighted.repeat(latent_space_.shape[0], 1).reshape(latent_space_.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        # blosum = self.embed(blosum)
        # latent_space_ = torch.cat((latent_space_, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]

        assert latent_space.shape == (self.n_internal,self.align_seq_len, self.z_dim)

        sampling_input = dict(
            latent_space=latent_space_3d,
            decoder_hidden = decoder_hidden,
            n_samples=n_samples
        )

        aa_sequences,logits = itemgetter("aa_sequences","logits")(self.sampling_categorical_likelihood(sampling_input)) # #EXCHANGEABLE MODULE

        sampling_out = SamplingOutput(aa_sequences=aa_sequences.detach(),
                                      latent_space=latent_space.detach(),
                                      logits=logits.detach(),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None,
                                      covariance=covariance
                                      )

        return sampling_out

class DRAUPNIRGUIDES(EasyGuide):

    def __init__(self,draupnir_model,ModelLoad, Draupnir):
        super(DRAUPNIRGUIDES, self).__init__(draupnir_model)
        self.guide_type = ModelLoad.args.select_guide
        self.draupnir = Draupnir
        self.args = ModelLoad.args
        self.encoder_input_size = self.draupnir.aa_probs
        self.dataset_train_blosum = self.draupnir.dataset_train_blosum
        self.batch_size = self.draupnir.batch_size #comes from build config
        self.args = self.draupnir.args
        self.device = self.draupnir.device
        self.encoder = None #fill in
        self.embeddingencoder = None # fill in


        if self.draupnir.args.covariance_prior == "5": #example with the whitening prior
            self.log_lambd = PyroParam(torch.tensor([0.]), event_dim=0)


    def get_class(self):
        full_name = self.__class__
        name = str(full_name).split(".")[-1].replace("'>","")
        return name

    def guide(self, datasets, patristic_matrix_train_sorted):
        """

        :param torch.tensor patristic_matrix_train_sorted: patristic matrix of shape (n_leaves + 1, n_leaves + 1), where the columns and the rows are the integer-encoded positions of the sequence in the tree (in tree leverl order)
        :param dict datasets: dictionary containing the integer-encoded sequences, the blosum encoded sequences and a few other things
        """


        pyro.module("encoder", self.encoder)
        pyro.module("embeddingsencoder", self.embeddingencoder)
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        n_train = aminoacid_sequences.shape[0]
        log_lambd = pyro.sample("log_lambd", dist.Delta(self.log_lambd).expand_by([1]))  # characteristic length-scale

        with pyro.plate("plate_batch", dim=-2, device=self.draupnir.device):
            aminoacid_sequences = self.embeddingencoder(datasets["blosum"])
            encoder_h_0 = self.h_0_GUIDE.expand(self.encoder.num_layers * 2, aminoacid_sequences.shape[0],self.draupnir.gru_hidden_dim).contiguous()
            encoder_output = self.encoder(aminoacid_sequences, encoder_h_0)  # [n,z_dim]
            assert isinstance(encoder_output,dict)
            z_loc, z_scale = encoder_output["z_loc"], encoder_output["z_scale"]
            assert z_loc.shape == (n_train, self.draupnir.z_dim)
            eps_z = pyro.sample("eps_z", dist.Normal(z_loc, z_scale).to_event(2))  # [n,z_dim]
            assert eps_z.shape == (n_train, self.draupnir.z_dim)

        return {
            "log_lambd": log_lambd.squeeze(),
            "lambd": torch.exp(log_lambd).squeeze(),
            "z_loc": z_loc,
            "z_scale": z_scale,
            "eps_z": eps_z,
            "node_names": datasets["int"][:, 0, 1]
        }






