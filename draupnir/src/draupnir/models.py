"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
import warnings
from collections import namedtuple
from abc import abstractmethod
import torch.nn as nn
import torch
import draupnir.utils as DraupnirUtils
from draupnir.models_utils import *
import pyro
import pyro.distributions as dist
from scipy import stats


SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi","covariance"])

class DRAUPNIRModelClass(nn.Module):
    def __init__(self, ModelLoad):
        super(DRAUPNIRModelClass, self).__init__()
        self.args = ModelLoad.args
        self.num_epochs = ModelLoad.args.num_epochs
        self.gru_hidden_dim = ModelLoad.gru_hidden_dim
        self.embedding_dim = ModelLoad.args.embedding_dim #blosum embedding dim
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
        self.input_size = self.z_dim
        self.use_attention = False
        self.batch_first = True
        self.leaves_testing = ModelLoad.leaves_testing
        self.batch_by_clade = ModelLoad.args.batch_by_clade
        self.device = ModelLoad.device
        self.kappa_addition = ModelLoad.args.kappa_addition
        self.aa_frequencies_train = ModelLoad.aa_frequencies_train
        self.blosum = ModelLoad.blosum
        self.blosum_max = ModelLoad.blosum_max
        self.blosum_weighted = ModelLoad.blosum_weighted #based solely on the train dataset
        self.dataset_train_blosum = ModelLoad.dataset_train_blosum
        self.variable_score = ModelLoad.variable_score
        self.internal_nodes = ModelLoad.internal_nodes
        self.batch_size = ModelLoad.build_config.batch_size
        self.plating = ModelLoad.args.plating
        self.plate_size = ModelLoad.build_config.plate_subsample_size
        self.plate_unordered = ModelLoad.plate_unordered
        self.one_hot_encoding = ModelLoad.one_hot_encoding
        self.n_leaves_batch = self.batch_size
        self.n_internal_batch = self.batch_size
        self.n_leaves = len(self.leaves_nodes)
        self.n_internal = len(self.internal_nodes)
        self.n_all = self.n_leaves + self.n_internal
        self.num_layers = 1
        self.tree_height = ModelLoad.tree_height
        self.covariance = None
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.gp_priors_experiments_dict = {"1":self.gp_prior_batched_experiment1,
                                           "2":self.gp_prior_batched_experiment2,
                                           "3":self.gp_prior_batched_experiment3,
                                           "4":self.gp_prior_batched_experiment4,
                                           "5":self.gp_prior_batched_experiment5
                                           }

        self.conditional_sampling_batch_dict = {"1":self.conditional_sampling_batch_experiment1,
                                                       "2": self.conditional_sampling_batch_experiment2,
                                                       "3": self.conditional_sampling_batch_experiment3,
                                                       "4": self.conditional_sampling_batch_experiment4,
                                                       "5": self.conditional_sampling_batch_experiment5,
                                                       }
        if self.args.use_cuda:
            self.cuda()
    @abstractmethod
    def guide(self, datasets, patristic_matrix, patristic_matrix_eval, data_blosum, batch_blosum,map_estimates):
        raise NotImplementedError
    @abstractmethod
    def model(self, datasets, patristic_matrix, patristic_matrix_eval, data_blosum, batch_blosum,map_estimates):
        raise NotImplementedError
    @abstractmethod
    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix, patristic_matrix_eval, use_test=True):
        raise NotImplementedError
    @abstractmethod
    def get_class(self):
        full_name = self.__class__
        name = str(full_name).split(".")[-1].replace("'>","")
        return name
    def gp_prior(self,patristic_matrix_sorted):
        """Computes an Ornstein Ulenbeck process prior over the latent space, representing the evolutionary process.
        The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances tu build a covariance matrix"""
        # Highlight; OU kernel parameters
        alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(1)) + 1e-6 #TODO: Change to another distribution less centered around 0
        sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.z_dim, ]).to_event(1)) +1e-6 # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
        sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand_by([self.z_dim, ]).to_event(1))  +1e-6 # Gaussian noise
        lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.z_dim, ]).to_event(1))  +1e-6 # characteristic length-scale

        alpha = DraupnirUtils.squeeze_tensor(1,alpha)
        sigma_f = DraupnirUtils.squeeze_tensor(1,sigma_f)
        sigma_n = DraupnirUtils.squeeze_tensor(1,sigma_n)
        lambd = DraupnirUtils.squeeze_tensor(1,lambd)

        # Highlight: Sample the latent space from MultivariateNormal with GP prior on covariance
        patristic_matrix = patristic_matrix_sorted[1:, 1:]
        OU_covariance = OUKernel_Fast(sigma_f, sigma_n, lambd).forward(patristic_matrix)
        OU_mean = torch.zeros((patristic_matrix.shape[0],)).unsqueeze(0)
        # print("Model Covariance: {}".format(OU_covariance.shape))
        # print("Model Mean: {}".format(OU_mean.shape))

        if self.leaves_testing:
            assert OU_covariance.shape == (self.z_dim, self.n_all, self.n_all),f"Expected shape {(self.z_dim, self.n_all, self.n_all)}, got {OU_covariance.shape}"
            assert OU_mean.shape == (1, self.n_all)
        else:

            assert OU_covariance.shape == (self.z_dim, self.n_leaves, self.n_leaves), f"Expected shape {(self.z_dim, self.n_leaves, self.n_leaves)}, got {OU_covariance.shape}"
            assert OU_mean.shape == (1,self.n_leaves)
        #noise = 1e-15 + torch.eye(OU_covariance.shape[1])
        #https://github.com/pyro-ppl/pyro/issues/702
        #https://forum.pyro.ai/t/runtimeerror-during-cholesky-decomposition/1216/2---> fix runtime error with choleky decomposition
        #https://forum.pyro.ai/t/using-constraints-within-an-nn-module/486
        #OU_covariance = transform_to(constraints.lower_cholesky)(OU_covariance) #check that this does not affect performance
        latent_space = pyro.sample('latent_z', dist.MultivariateNormal(OU_mean, OU_covariance ).to_event(1)) #[z_dim=30,n_nodes] #+ noise[None,:,:]
        #print("Model Latent space: {}".format(latent_space.shape))
        latent_space = latent_space.T
        return {"latent_space": latent_space,"covariance": OU_covariance}
    def gp_prior_experiment5(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"
        patristic_matrix = patristic_matrix_sorted[1:, 1:]  # [n_leaves_batch,n_leaves_batch]

        D_max = patristic_matrix.max()
        log_lambd = pyro.sample("log_lambda", dist.Normal(torch.log(D_max / 2), 1.0).expand_by([1]))
        lambd = torch.exp(log_lambd)

        lambd = DraupnirUtils.squeeze_tensor(1,lambd)

        OU_covariance = OUKernel_Fast_experiment(None, lambd, None).forward(patristic_matrix)  # [n_leaves,n_leaves ]
        OU_covariance = DraupnirUtils.squeeze_tensor(2, OU_covariance)

        assert OU_covariance.shape == (self.n_leaves,self.n_leaves), f"Expected shape: ({self.n_leaves},{self.n_leaves}), got ({OU_covariance.shape})"
        L = torch.linalg.cholesky(OU_covariance)
        eps_z = pyro.sample("eps_z", dist.Normal(0, 1).expand_by([self.n_leaves, self.z_dim]))  # adds some noise to each of the leaves?
        latent_space = L @ eps_z

        assert latent_space.shape == (self.n_leaves,self.z_dim)

        return {"latent_space": latent_space,"covariance": OU_covariance}

    def gp_prior_batched(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"
        # Highlight; OU kernel parameters #TODO: Add noise to OU parameters to avoid error in cholesky decomposition
        alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(0))
        sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.z_dim, ]).to_event(0))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
        sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand_by([self.z_dim, ]).to_event(0))  # Gaussian noise
        lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.z_dim, ]).to_event(0))  # characteristic length-scale

        sigma_f = DraupnirUtils.squeeze_tensor(1, sigma_f) + 1e-6 #TODO: Cannot make the OU process parameters squeeze
        sigma_n = DraupnirUtils.squeeze_tensor(1, sigma_n) + 1e-6
        lambd = DraupnirUtils.squeeze_tensor(1, lambd) + 1e-6

        # Highlight: Sample the latent space from MultivariateNormal with GP prior on covariance
        patristic_matrix = patristic_matrix_sorted[1:, 1:]

        OU_covariance = OUKernel_Fast(sigma_f, sigma_n, lambd).forward(patristic_matrix)
        OU_mean = torch.zeros((patristic_matrix.shape[0],)).unsqueeze(0)

        assert OU_covariance.shape == (self.z_dim, self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        assert OU_mean.shape == (1,self.n_leaves_batch)
        #noise = 1e-15 + torch.eye(OU_covariance.shape[1])
        #https://github.com/pyro-ppl/pyro/issues/702
        #https://forum.pyro.ai/t/runtimeerror-during-cholesky-decomposition/1216/2---> fix runtime error with choleky decomposition
        #https://forum.pyro.ai/t/using-constraints-within-an-nn-module/486
        #OU_covariance = transform_to(constraints.lower_cholesky)(OU_covariance) #check that this does not affect performance
        latent_space = pyro.sample('latent_z', dist.MultivariateNormal(OU_mean, OU_covariance ).to_event(1)) #[z_dim=30,n_nodes] #+ noise[None,:,:]
        latent_space = latent_space.T
        return {"latent_space": latent_space, "covariance": OU_covariance}
    def gp_prior_batched_experiment1(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"
        # Highlight; OU kernel parameters #TODO: Add noise to OU parameters to avoid error in cholesky decomposition

        sigma_f = pyro.sample("sigma_f",dist.HalfNormal(1).expand_by([self.z_dim])) + 1e-6
        log_lambd = pyro.sample("lambd",dist.Normal(torch.log(self.tree_height/2),0.5).expand_by([self.z_dim]).to_event(0)) #assuming neperian logarithm
        lambd = torch.exp(log_lambd)

        sigma_f = DraupnirUtils.squeeze_tensor(1, sigma_f) + 1e-6 #TODO: Cannot make the OU process parameters squeeze
        lambd = DraupnirUtils.squeeze_tensor(1, lambd) + 1e-6

        # Highlight: Sample the latent space from MultivariateNormal with GP prior on covariance
        patristic_matrix = patristic_matrix_sorted[1:, 1:]
        OU_covariance = OUKernel_Fast_experiment(sigma_f, lambd, None).forward(patristic_matrix)
        OU_mean = torch.zeros((patristic_matrix.shape[0],)).unsqueeze(0)

        assert OU_covariance.shape == (self.z_dim, self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        assert OU_mean.shape == (1,self.n_leaves_batch)
        #noise = 1e-15 + torch.eye(OU_covariance.shape[1])
        #https://github.com/pyro-ppl/pyro/issues/702
        #https://forum.pyro.ai/t/runtimeerror-during-cholesky-decomposition/1216/2---> fix runtime error with choleky decomposition
        #https://forum.pyro.ai/t/using-constraints-within-an-nn-module/486
        #OU_covariance = transform_to(constraints.lower_cholesky)(OU_covariance) #check that this does not affect performance
        latent_space = pyro.sample('latent_z', dist.MultivariateNormal(OU_mean, OU_covariance ).to_event(1)) #[z_dim=30,n_nodes] #+ noise[None,:,:]
        latent_space = latent_space.T
        return {"latent_space": latent_space,"covariance": OU_covariance}
    def gp_prior_batched_experiment2(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"
        # Highlight; OU kernel parameters #TODO: Add noise to OU parameters to avoid error in cholesky decomposition

        alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(0))
        sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([1, ]).to_event(0))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
        sigma_n = pyro.sample("sigma_n",dist.HalfNormal(alpha[1]).expand_by([1, ]).to_event(0))  # Gaussian noise
        lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([1, ]).to_event(0))  # characteristic length-scale

        sigma_f = DraupnirUtils.squeeze_tensor(1, sigma_f).repeat(self.z_dim) + 1e-6 #TODO: Cannot make the OU process parameters squeeze
        sigma_n = DraupnirUtils.squeeze_tensor(1, sigma_n).repeat(self.z_dim) + 1e-6 #TODO: Cannot make the OU process parameters squeeze
        lambd = DraupnirUtils.squeeze_tensor(1, lambd).repeat(self.z_dim) + 1e-6

        # Highlight: Sample the latent space from MultivariateNormal with GP prior on covariance
        patristic_matrix = patristic_matrix_sorted[1:, 1:]
        OU_covariance = OUKernel_Fast_experiment(sigma_f, lambd, sigma_n).forward(patristic_matrix)
        OU_mean = torch.zeros((patristic_matrix.shape[0],)).unsqueeze(0)

        assert OU_covariance.shape == (self.z_dim, self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        assert OU_mean.shape == (1,self.n_leaves_batch)
        #noise = 1e-15 + torch.eye(OU_covariance.shape[1])
        #https://github.com/pyro-ppl/pyro/issues/702
        #https://forum.pyro.ai/t/runtimeerror-during-cholesky-decomposition/1216/2---> fix runtime error with choleky decomposition
        #https://forum.pyro.ai/t/using-constraints-within-an-nn-module/486
        #OU_covariance = transform_to(constraints.lower_cholesky)(OU_covariance) #check that this does not affect performance
        latent_space = pyro.sample('latent_z', dist.MultivariateNormal(OU_mean, OU_covariance ).to_event(1)) #[z_dim=30,n_nodes] #+ noise[None,:,:]
        latent_space = latent_space.T
        return {"latent_space": latent_space,"covariance": OU_covariance}
    def gp_prior_batched_experiment3(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"

        rho = pyro.sample("rho", dist.Beta(8,2).expand_by([self.z_dim])) + 1e-6 # correlation prior
        lambd = -2/torch.log(rho)

        #po = DraupnirUtils.squeeze_tensor(1, po) + 1e-6
        lambd = DraupnirUtils.squeeze_tensor(1, lambd) + 1e-6

        # Highlight: Sample the latent space from MultivariateNormal with GP prior on covariance
        patristic_matrix = patristic_matrix_sorted[1:, 1:]
        OU_covariance = OUKernel_Fast_experiment(None, lambd, None).forward(patristic_matrix)
        OU_mean = torch.zeros((patristic_matrix.shape[0],)).unsqueeze(0)

        assert OU_covariance.shape == (self.z_dim, self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        assert OU_mean.shape == (1,self.n_leaves_batch)
        #noise = 1e-15 + torch.eye(OU_covariance.shape[1])
        #https://github.com/pyro-ppl/pyro/issues/702
        #https://forum.pyro.ai/t/runtimeerror-during-cholesky-decomposition/1216/2---> fix runtime error with choleky decomposition
        #https://forum.pyro.ai/t/using-constraints-within-an-nn-module/486
        #OU_covariance = transform_to(constraints.lower_cholesky)(OU_covariance) #check that this does not affect performance
        latent_space = pyro.sample('latent_z', dist.MultivariateNormal(OU_mean, OU_covariance ).to_event(1)) #[z_dim=30,n_nodes] #+ noise[None,:,:]
        latent_space = latent_space.T
        return {"latent_space": latent_space,"covariance": OU_covariance}
    def gp_prior_batched_experiment4(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"


        patristic_matrix = patristic_matrix_sorted[1:, 1:]  # [n_leaves_batch,n_leaves_batch]
        rho = pyro.sample("rho", dist.Beta(8, 2).expand_by([1])) + 1e-6  # correlation prior
        rho = rho.clamp(1e-8, 1 - 1e-8)
        lambd = -2/torch.log(rho)
        rho = DraupnirUtils.squeeze_tensor(1, rho)

        OU_covariance = OUKernel_Fast_experiment(None, lambd, None).forward(patristic_matrix)  # [n_leaves,n_leaves ]
        OU_covariance = DraupnirUtils.squeeze_tensor(2, OU_covariance)



        assert OU_covariance.shape == (self.n_leaves_batch,self.n_leaves_batch), f"Expected shape: ({self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        L = torch.linalg.cholesky(OU_covariance)
        eps_z = pyro.sample("eps_z", dist.Normal(0, 1).expand_by([self.n_leaves_batch, self.z_dim]))  # adds some noise to each of the leaves?
        latent_space = L @ eps_z


        assert latent_space.shape == (self.n_leaves_batch,self.z_dim)

        return {"latent_space": latent_space,"covariance": OU_covariance}
    def gp_prior_batched_experiment5(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"
        patristic_matrix = patristic_matrix_sorted[1:, 1:]  # [n_leaves_batch,n_leaves_batch]

        D_max = patristic_matrix.max()
        log_lambd = pyro.sample("log_lambda", dist.Normal(torch.log(D_max / 2), 1.0).expand_by([1]))
        lambd = torch.exp(log_lambd)

        lambd = DraupnirUtils.squeeze_tensor(1,lambd)

        OU_covariance = OUKernel_Fast_experiment(None, lambd, None).forward(patristic_matrix)  # [n_leaves,n_leaves ]
        OU_covariance = DraupnirUtils.squeeze_tensor(2, OU_covariance)

        assert OU_covariance.shape == (self.n_leaves_batch,self.n_leaves_batch), f"Expected shape: ({self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        L = torch.linalg.cholesky(OU_covariance)
        eps_z = pyro.sample("eps_z", dist.Normal(0, 1).expand_by([self.n_leaves_batch, self.z_dim]))  # adds some noise to each of the leaves?
        latent_space = L @ eps_z

        assert latent_space.shape == (self.n_leaves_batch,self.z_dim)

        return {"latent_space": latent_space,"covariance": OU_covariance}
    def prediction_batching_preprocessing(self,map_estimates,patristic_matrix_full,patristic_matrix_test,batch_idx,use_test,use_test2):
        """Correction of a few parameters to be able to carry on with the batched sampling"""
        if use_test or use_test2:# internal nodes. Only Marginal posterior available when batching
            assert patristic_matrix_full[1:,1:].shape == (self.n_all,self.n_all)
            #Highlight: Slice out the train sequences and only a batch from the test sequences
            if batch_idx[1] is None:
                self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0]) + 1:, 0]
            else:
                self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0])+1:int(batch_idx[1])+1,0]
            self.n_internal_batch = len(self.internal_nodes_batch)
            self.leaves_nodes = map_estimates["train_leaves_nodes"] if "train_leaves_nodes" in map_estimates.keys() else self.leaves_nodes
            self.n_leaves = len(self.leaves_nodes)
            nodes_batch = torch.cat((self.leaves_nodes,self.internal_nodes_batch)) #this needs to contain only the leave nodes on
            self.n_leaves_internal_batch = len(nodes_batch) #leave nodes + internal nodes
            indexes = (patristic_matrix_full[:, 0][..., None] == nodes_batch).any(-1)
            indexes[0] = True #re-add the nodes names
            # patristic_matrix = patristic_matrix_full[indexes]
            # patristic_matrix = patristic_matrix[:,indexes]
            # cond_samp_out_dict = self.conditional_sampling_batch(map_estimates,patristic_matrix)
            patristic_matrix_test_batch = patristic_matrix_full[indexes]
            patristic_matrix_test_batch = patristic_matrix_test_batch[:,indexes]
            cond_samp_out_dict = self.conditional_sampling_batch(map_estimates,patristic_matrix_test_batch)
            latent_space = cond_samp_out_dict["latent_space"]

            covariance = cond_samp_out_dict["covariance"]
            if covariance.ndim == 2:
                covariance = covariance[cond_samp_out_dict["internal_idx"]] #[n_test_batch, n_test_batch+n_train]
                covariance = covariance[:,cond_samp_out_dict["internal_idx"]] #[n_test_batch, n_test_batch]
            else:
                covariance = covariance[:,cond_samp_out_dict["internal_idx"]] #[n_test_batch, n_test_batch+n_train]
                covariance = covariance[:,:,cond_samp_out_dict["internal_idx"]] #[n_test_batch, n_test_batch]

            n_nodes = self.n_internal_batch

        else: #training/leaves

            n_nodes = self.n_leaves_batch #here n_leaves has been overloaded by the batch size
            latent_space = map_estimates["latent_z"].T #the map estimates have been pre-concatenated, that is why we have to index them out
            latent_space = latent_space[int(batch_idx[0]):int(batch_idx[1])] if batch_idx is not None else latent_space

            # if batch_idx is not None:  # if it is None then the shape should be correct already (for the test_batched_train_batched approach)
            #     if self.covariance.ndim == 2:
            #         covariance = self.covariance[batch_idx[0]:batch_idx[1]] if batch_idx[1] is not None else self.covariance[batch_idx[0]:] # this should be in the same order as the predicted dataset (we override the self.covariance when we predict)
            #     else:
            #         print("BEFORE !prediction batching preprocessing ", self.covariance.shape)
            #
            #         print(batch_idx[0],batch_idx[1])
            #
            #         covariance = self.covariance[:,batch_idx[0]:batch_idx[1]] if batch_idx[1] is not None else self.covariance[:,batch_idx[0]:batch_idx[1]]
            #
            #         print("AFTER !prediction batching preprocessing ", covariance.shape)
            # else:
            covariance = self.covariance #i think here we are returning the covariance from the last batch over and over -> TODO: Stack somehow the covariances ? then the indexing will make sense

            assert latent_space.shape == (n_nodes, self.z_dim)

        return {"latent_space": latent_space, "n_nodes": n_nodes, "covariance": covariance}

    def prediction_batching_preprocessing_experiment(self,map_estimates,patristic_matrix_full,patristic_matrix_test,batch_idx,use_test,use_test2):
        """Correction of a few parameters to be able to carry on with the batched sampling"""
        self.leaves_nodes = map_estimates["train_leaves_nodes"] if "train_leaves_nodes" in map_estimates.keys() else self.leaves_nodes
        if use_test or use_test2:# internal nodes. Only Marginal posterior available when batching
            assert patristic_matrix_full[1:,1:].shape == (self.n_all,self.n_all)
            #Highlight: Slice out the train sequences and only a batch from the test sequences
            if batch_idx[1] is None:
                self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0]) + 1:, 0]
            else:
                self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0])+1:int(batch_idx[1])+1,0]
            self.n_internal_batch = len(self.internal_nodes_batch)
            self.n_leaves = len(self.leaves_nodes)
            nodes_batch = torch.cat((self.leaves_nodes,self.internal_nodes_batch)) #this needs to contain only the leave nodes on
            self.n_leaves_internal_batch = len(nodes_batch) #leave nodes + internal nodes
            indexes = (patristic_matrix_full[:, 0][..., None] == nodes_batch).any(-1)
            indexes[0] = True #re-add the nodes names
            patristic_matrix_batch = patristic_matrix_full[indexes] # all leaves + batch internal
            patristic_matrix_batch = patristic_matrix_batch[:,indexes]
            cond_samp_out_dict = self.conditional_sampling_batch(map_estimates, patristic_matrix_batch)
            latent_space = cond_samp_out_dict["latent_space"]
            covariance = cond_samp_out_dict["covariance"][cond_samp_out_dict["internal_idx"]] #[n_test_batch, n_test_batch+n_train]
            covariance = covariance[:,cond_samp_out_dict["internal_idx"]] #[n_test_batch, n_test_batch]
            n_nodes = self.n_internal_batch

        else: #training/leaves
            n_nodes = self.n_leaves_batch #here n_leaves has been overloaded by the batch size

            if "latent_z" in map_estimates.keys():
                latent_space = map_estimates["latent_z"].T
            elif "eps_z" in map_estimates.keys():
                idx_train = (patristic_matrix_full[:,0][...,None] == self.leaves_nodes).any(-1)
                idx_train[0] = True
                patristic_matrix_train = patristic_matrix_full[idx_train]
                patristic_matrix_train = patristic_matrix_train[:,idx_train]
                patristic_matrix_train = patristic_matrix_train[1:,1:] #remember to remove the node names


                if self.args.prior_experiment == "4":
                    lambd = -2/torch.log(map_estimates["rho"])

                elif self.args.prior_experiment == "5":
                    lambd = torch.exp(map_estimates["log_lambd"])

                covariance  = OUKernel_Fast_experiment(None, lambd, None).forward(patristic_matrix_train)

                L = torch.linalg.cholesky(covariance)
                latent_space = L @ map_estimates["eps_z"]

                if batch_idx is not None: #if it is None then the shape should be correct already
                    if covariance.ndim == 2:
                        covariance = covariance[batch_idx[0]:batch_idx[1]] if batch_idx[1] is not None else covariance[batch_idx[0]:]  # this should be in the same order as the predicted dataset (we override the self.covariance when we predict)
                    else:
                        covariance = covariance[:, batch_idx[0]:batch_idx[1]] if batch_idx[1] is not None else covariance[:, batch_idx[0]:]  # this should be in the same order  as the map estimates


            latent_space = latent_space[int(batch_idx[0]):int(batch_idx[1])] if batch_idx is not None else latent_space
            assert latent_space.shape == (n_nodes, self.z_dim)

        return {"latent_space": latent_space, "n_nodes": n_nodes, "covariance": covariance}

    def map_sampling(self,map_estimates,patristic_matrix_full):
        "Use map sampling for leaves prediction/testing, when internal nodes are not available"

        warnings.warn("This needs to be fixed")

        test_indexes = (patristic_matrix_full[1:, 0][..., None] == self.internal_nodes).any(-1) #indexes of the leaves selected for testing
        latent_space_internal = map_estimates["latent_z"][:, test_indexes].T
        assert latent_space_internal.shape == (self.n_internal, self.z_dim)
        return latent_space_internal

    def conditional_sampling(self,map_estimates, patristic_matrix):
            """Conditional sampling the internal nodes given the leaves from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)
            :param map_estimates: dictionary conatining the MAP estimates for the OU process parameters
            :param patristic_matrix: full patristic matrix"""
            sigma_f = DraupnirUtils.squeeze_tensor(1,map_estimates["sigma_f"])
            sigma_n = DraupnirUtils.squeeze_tensor(1,map_estimates["sigma_n"])
            lambd = DraupnirUtils.squeeze_tensor(1,map_estimates["lambd"])

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
            assert inverse_full.shape == (self.z_dim, self.n_all, self.n_all), f"Expected dimensions : {(self.z_dim, self.n_all, self.n_all)}, got {inverse_full.shape}"
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
            inverse_internal_bis = torch.linalg.inv(inverse_internal) #https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            # solve A@A-1 = I with torch.linalg.solve to have a faster and more stable calculation. torch.linal.solve calculates X from the A@X= B equation, here A is the inverse_internal, B is the Identity function
            # X will be the A-1
            # internal_identity = torch.eye(inverse_internal.size(1)).repeat(inverse_internal.size(0), 1, 1)
            # inverse_internal_bis = torch.linalg.solve(inverse_internal, internal_identity)

            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal)

            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis + 1e-6).to_event(1).sample()
            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal, self.z_dim)

            return {"latent_space": latent_space,"covariance": OU_covariance_full, "internal_idx": internal_indexes}

    def conditional_sampling_experiment5(self,map_estimates, patristic_matrix):
            """Conditional sampling the internal nodes given the leaves from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)
            :param map_estimates: dictionary conatining the MAP estimates for the OU process parameters
            :param patristic_matrix: full patristic matrix"""

            log_lambd = map_estimates["log_lambd"] #+ 1e-6
            lambd = torch.exp(log_lambd)
            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes).any(-1)
            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditioned on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_full = patristic_matrix[1:, 1:]
            assert patristic_matrix_full.shape == (self.n_all, self.n_all), "Remember to use the entire/full patristic matrix for conditional sampling!"
            OU = OUKernel_Fast_experiment(None,lambd,None)
            OU_covariance_full = OU.forward(patristic_matrix_full)
            L_full = torch.linalg.cholesky(OU_covariance_full)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.n_all, self.n_all), f"Expected dimensions : {( self.n_all, self.n_all)}, got {inverse_full.shape}"
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[internal_indexes, :]
            inverse_internal = inverse_internal[:, internal_indexes]  # [z_dim,n_test,n_test]
            assert inverse_internal.shape == (self.n_internal, self.n_internal)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:, ~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.n_internal, self.n_leaves)
            # Highlight: xb

            eps_z = map_estimates["eps_z"]
            L_train = L_full[~internal_indexes]
            L_train = L_train[:,~internal_indexes]
            xb = (L_train@eps_z).T #[zdim, ntrain]

            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            inverse_internal_bis = torch.linalg.inv(inverse_internal) #https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            # solve A@A-1 = I with torch.linalg.solve to have a faster and more stable calculation. torch.linal.solve calculates X from the A@X= B equation, here A is the inverse_internal, B is the Identity function
            # X will be the A-1
            # internal_identity = torch.eye(inverse_internal.size(1)).repeat(inverse_internal.size(0), 1, 1)
            # inverse_internal_bis = torch.linalg.solve(inverse_internal, internal_identity)

            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # [z_dim,n_test,n_train]

            assert part1.shape == (self.n_internal, self.n_leaves)
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal)

            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis + 1e-6).to_event(1).sample()
            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal, self.z_dim)

            return {"latent_space": latent_space,"covariance": OU_covariance_full, "internal_idx": internal_indexes}

    def conditional_samplingMAP(self,map_estimates, patristic_matrix):
            """Conditional sampling the internal nodes given the leaves from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)
            :param map_estimates: dictionary conatining the MAP estimates for the OU process parameters
            :param patristic_matrix: full patristic matrix"""
            sigma_f = DraupnirUtils.squeeze_tensor(1, map_estimates["sigma_f"])
            sigma_n = DraupnirUtils.squeeze_tensor(1, map_estimates["sigma_n"])
            lambd = DraupnirUtils.squeeze_tensor(1, map_estimates["lambd"])
            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes).any(-1)

            #n_internal = datasets_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_full = patristic_matrix[1:, 1:]
            assert patristic_matrix_full.shape == (self.n_all, self.n_all), "Remember to use the entire/full patristic matrix for conditional sampling!"
            OU = OUKernel_Fast(sigma_f, sigma_n, lambd)
            OU_covariance_full = OU.forward(patristic_matrix_full)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.z_dim, self.n_all, self.n_all), f"Expected dimensions : {(self.z_dim, self.n_all, self.n_all)}, got {inverse_full.shape}"
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
            inverse_internal_bis = torch.linalg.inv(inverse_internal) #https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            # solve A@A-1 = I with torch.linalg.solve to have a faster and more stable calculation. torch.linal.solve calculates X from the A@X= B equation, here A is the inverse_internal, B is the Identity function
            # X will be the A-1
            # internal_identity = torch.eye(inverse_internal.size(1)).repeat(inverse_internal.size(0), 1, 1)
            # inverse_internal_bis = torch.linalg.solve(inverse_internal, internal_identity)

            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal)
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis).to_event(1).sample()
            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal, self.z_dim)

            return {"latent_space": OU_mean.squeeze(-1).T, "covariance": OU_covariance_full, "internal_idx": internal_indexes}

    def conditional_sampling_batch(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""
            sigma_f = DraupnirUtils.squeeze_tensor(1, map_estimates["sigma_f"]) + 1e-6
            sigma_n = DraupnirUtils.squeeze_tensor(1, map_estimates["sigma_n"])  + 1e-6
            lambd = DraupnirUtils.squeeze_tensor(1, map_estimates["lambd"]) + 1e-6
            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes_batch).any(-1)

            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_batch = patristic_matrix[1:, 1:]
            assert patristic_matrix_batch.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch), "Here we are using a slice of the patristic matrix with size n_leaves_batch = batch_size!"
            OU = OUKernel_Fast(sigma_f, sigma_n, lambd)
            OU_covariance_full = OU.forward(patristic_matrix_batch)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.z_dim, self.n_leaves_internal_batch, self.n_leaves_internal_batch)
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[:, internal_indexes, :]
            inverse_internal = inverse_internal[:, :, internal_indexes]  # [z_dim,n_test,n_test]
            assert inverse_internal.shape == (self.z_dim, self.n_internal_batch, self.n_internal_batch)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal_batch,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:, :, ~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.z_dim, self.n_internal_batch, self.n_leaves)
            # Highlight: xb
            xb = map_estimates["latent_z"]  # [z_dim,n_train] # ok, so we need to get the map estimates for all the train latents
            # if self.leaves_testing:
            #     leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1) #only the indexes of the training leaves
            #     xb = xb[:,leaves_indexes]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            inverse_internal_bis = torch.linalg.inv(inverse_internal) #https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            #solve A@A-1 = I with torch.linalg.solve to have a faster and more stable calculation. torch.linal.solve calculates X from the A@X= B equation, here A is the inverse_internal, B is the Identity function
            # X will be the A-1
            # internal_identity = torch.eye(inverse_internal.size(1)).repeat(inverse_internal.size(0),1,1)
            # inverse_internal_bis = torch.linalg.solve(inverse_internal,internal_identity)
            # residual = torch.matmul(inverse_internal, inverse_internal_bis) - internal_identity  # should be close to zero
            # max_err = residual.abs().max()
            # print(f"max err: {max_err}")
            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal_batch)
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis).to_event(1).sample()
            #latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.cholesky_inverse(Inverse_internal) + 1e-6).to_event(1).sample()

            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal_batch, self.z_dim)
            return {"latent_space": latent_space,"covariance": OU_covariance_full, "internal_idx": internal_indexes}

    def conditional_sampling_batch_experiment1(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""


            sigma_f = DraupnirUtils.squeeze_tensor(1, map_estimates["sigma_f"]) + 1e-6
            lambd = DraupnirUtils.squeeze_tensor(1, map_estimates["lambd"]) + 1e-6
            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes_batch).any(-1)

            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_batch = patristic_matrix[1:, 1:]
            assert patristic_matrix_batch.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch), "Here we are using a slice of the patristic matrix with size n_leaves_batch = batch_size!"

            OU = OUKernel_Fast_experiment(sigma_f, lambd, None)
            OU_covariance_full = OU.forward(patristic_matrix_batch)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.z_dim, self.n_leaves_internal_batch, self.n_leaves_internal_batch)
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[:, internal_indexes, :]
            inverse_internal = inverse_internal[:, :, internal_indexes]  # [z_dim,n_test,n_test]
            assert inverse_internal.shape == (self.z_dim, self.n_internal_batch, self.n_internal_batch)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal_batch,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:, :, ~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.z_dim, self.n_internal_batch, self.n_leaves)
            # Highlight: xb
            xb = map_estimates["latent_z"]  # [z_dim,n_train] # ok, so we need to get the map estimates for all the train latents
            # if self.leaves_testing:
            #     leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1) #only the indexes of the training leaves
            #     xb = xb[:,leaves_indexes]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            inverse_internal_bis = torch.linalg.inv(inverse_internal) #https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            #solve A@A-1 = I with torch.linalg.solve to have a faster and more stable calculation. torch.linal.solve calculates X from the A@X= B equation, here A is the inverse_internal, B is the Identity function
            # X will be the A-1
            # internal_identity = torch.eye(inverse_internal.size(1)).repeat(inverse_internal.size(0),1,1)
            # inverse_internal_bis = torch.linalg.solve(inverse_internal,internal_identity)
            # residual = torch.matmul(inverse_internal, inverse_internal_bis) - internal_identity  # should be close to zero
            # max_err = residual.abs().max()
            # print(f"max err: {max_err}")
            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal_batch)
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis).to_event(1).sample()
            #latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.cholesky_inverse(Inverse_internal) + 1e-6).to_event(1).sample()

            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal_batch, self.z_dim)
            return {"latent_space": latent_space,"covariance": OU_covariance_full, "internal_idx": internal_indexes}

    def conditional_sampling_batch_experiment2(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""


            sigma_f = DraupnirUtils.squeeze_tensor(1, map_estimates["sigma_f"]) + 1e-6
            sigma_n = DraupnirUtils.squeeze_tensor(1, map_estimates["sigma_n"]) + 1e-6
            lambd = DraupnirUtils.squeeze_tensor(1, map_estimates["lambd"]) + 1e-6

            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes_batch).any(-1)
            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_batch = patristic_matrix[1:, 1:]
            assert patristic_matrix_batch.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch), "Here we are using a slice of the patristic matrix with size n_leaves_batch = batch_size!"

            OU = OUKernel_Fast_experiment(sigma_f, lambd, sigma_n)
            OU_covariance_full = OU.forward(patristic_matrix_batch)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.z_dim, self.n_leaves_internal_batch, self.n_leaves_internal_batch)
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[:, internal_indexes, :]
            inverse_internal = inverse_internal[:, :, internal_indexes]  # [z_dim,n_test,n_test]
            assert inverse_internal.shape == (self.z_dim, self.n_internal_batch, self.n_internal_batch)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal_batch,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:, :, ~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.z_dim, self.n_internal_batch, self.n_leaves)
            # Highlight: xb
            xb = map_estimates["latent_z"]  # [z_dim,n_train] # ok, so we need to get the map estimates for all the train latents
            # if self.leaves_testing:
            #     leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1) #only the indexes of the training leaves
            #     xb = xb[:,leaves_indexes]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            inverse_internal_bis = torch.linalg.inv(inverse_internal) #https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            #solve A@A-1 = I with torch.linalg.solve to have a faster and more stable calculation. torch.linal.solve calculates X from the A@X= B equation, here A is the inverse_internal, B is the Identity function
            # X will be the A-1
            # internal_identity = torch.eye(inverse_internal.size(1)).repeat(inverse_internal.size(0),1,1)
            # inverse_internal_bis = torch.linalg.solve(inverse_internal,internal_identity)
            # residual = torch.matmul(inverse_internal, inverse_internal_bis) - internal_identity  # should be close to zero
            # max_err = residual.abs().max()
            # print(f"max err: {max_err}")
            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal_batch)
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis).to_event(1).sample()
            #latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.cholesky_inverse(Inverse_internal) + 1e-6).to_event(1).sample()

            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal_batch, self.z_dim)
            return {"latent_space": latent_space,"covariance": OU_covariance_full, "internal_idx": internal_indexes}

    def conditional_sampling_batch_experiment3(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""
            rho = DraupnirUtils.squeeze_tensor(1, map_estimates["rho"]) + 1e-6
            lambd = -2 / torch.log(rho)

            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes_batch).any(-1)
            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_batch = patristic_matrix[1:, 1:]
            assert patristic_matrix_batch.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch), "Here we are using a slice of the patristic matrix with size n_leaves_batch = batch_size!"

            OU = OUKernel_Fast_experiment(None, lambd, None)
            OU_covariance_full = OU.forward(patristic_matrix_batch)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.z_dim, self.n_leaves_internal_batch, self.n_leaves_internal_batch)
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[:, internal_indexes, :]
            inverse_internal = inverse_internal[:, :, internal_indexes]  # [z_dim,n_test,n_test]
            assert inverse_internal.shape == (self.z_dim, self.n_internal_batch, self.n_internal_batch)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal_batch,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[:,internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:, :, ~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.z_dim, self.n_internal_batch, self.n_leaves)
            # Highlight: xb
            xb = map_estimates["latent_z"]  # [z_dim,n_train] # ok, so we need to get the map estimates for all the train latents
            # if self.leaves_testing:
            #     leaves_indexes = (patristic_matrix[1:, 0][..., None] == self.leaves_nodes).any(-1) #only the indexes of the training leaves
            #     xb = xb[:,leaves_indexes]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            inverse_internal_bis = torch.linalg.inv(inverse_internal) #https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            #solve A@A-1 = I with torch.linalg.solve to have a faster and more stable calculation. torch.linal.solve calculates X from the A@X= B equation, here A is the inverse_internal, B is the Identity function
            # X will be the A-1
            # internal_identity = torch.eye(inverse_internal.size(1)).repeat(inverse_internal.size(0),1,1)
            # inverse_internal_bis = torch.linalg.solve(inverse_internal,internal_identity)
            # residual = torch.matmul(inverse_internal, inverse_internal_bis) - internal_identity  # should be close to zero
            # max_err = residual.abs().max()
            # print(f"max err: {max_err}")
            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # [z_dim,n_test,n_train]
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal_batch)
            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis).to_event(1).sample()
            #latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.cholesky_inverse(Inverse_internal) + 1e-6).to_event(1).sample()

            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal_batch, self.z_dim)

            return {"latent_space": latent_space,"covariance": OU_covariance_full, "internal_idx": internal_indexes}

    def conditional_sampling_batch_experiment4(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""
            rho = map_estimates["rho"] + 1e-6
            rho = rho.clamp(1e-8,1-1e-8)
            lambd = -2 / torch.log(rho)

            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes_batch).any(-1)
            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_batch = patristic_matrix[1:, 1:] #remove the node names

            assert patristic_matrix_batch.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch), "Here we are using a slice of the patristic matrix with size n_leaves_batch = batch_size!"
            OU = OUKernel_Fast_experiment(None, lambd, None)
            OU_covariance_full = OU.forward(patristic_matrix_batch) #+ torch.eye(patristic_matrix_batch.shape[0])*1e-6

            print(torch.diagonal(OU_covariance_full))

            if torch.allclose(OU_covariance_full, OU_covariance_full.T, atol=1e-6):
                print("matrix is symmetric")
            else:
                print("matrix is not symmetric")
            eigvals = torch.linalg.eigvalsh(OU_covariance_full)
            print("min eigenvalue per batch", eigvals.min(dim=-1).values)  # have to be > 0


            L_full = torch.linalg.cholesky(OU_covariance_full)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch)
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[internal_indexes]
            inverse_internal = inverse_internal[:, internal_indexes]  # [z_dim,n_test,n_test]
            #assert inverse_internal.shape == (self.z_dim, self.n_internal_batch, self.n_internal_batch)
            assert inverse_internal.shape == (self.n_internal_batch, self.n_internal_batch)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal_batch,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:,~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.n_internal_batch, self.n_leaves)
            # Highlight: xb
            eps_z = map_estimates["eps_z"]
            L_train = L_full[~internal_indexes]
            L_train = L_train[:,~internal_indexes]
            xb = (L_train@eps_z).T #[zdim, ntrain]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            inverse_internal_bis = torch.linalg.inv(inverse_internal)#https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # [z_dim,n_test,n_train]
            assert part1.shape == (self.n_internal_batch,self.n_leaves)
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :,None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal_batch)

            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis).to_event(1).sample()
            #latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.cholesky_inverse(Inverse_internal) + 1e-6).to_event(1).sample()

            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal_batch, self.z_dim)
            return {"latent_space": latent_space,"covariance": OU_covariance_full, "internal_idx": internal_indexes}

    def conditional_sampling_batch_experiment5(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""
            log_lambd = map_estimates["log_lambd"] #+ 1e-6
            lambd = torch.exp(log_lambd)

            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes_batch).any(-1)
            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditioned on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_batch = patristic_matrix[1:, 1:] #remove the node names

            assert patristic_matrix_batch.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch), "Here we are using a slice of the patristic matrix with size n_leaves_batch = batch_size!"
            OU = OUKernel_Fast_experiment(None, lambd, None)
            OU_covariance_full = OU.forward(patristic_matrix_batch) #+ torch.eye(patristic_matrix_batch.shape[0])*1e-6

            # print(torch.diagonal(OU_covariance_full))
            #
            # if torch.allclose(OU_covariance_full, OU_covariance_full.T, atol=1e-6):
            #     print("matrix is symmetric")
            # else:
            #     print("matrix is not symmetric")
            # eigvals = torch.linalg.eigvalsh(OU_covariance_full)
            # print("min eigenvalue per batch", eigvals.min(dim=-1).values)  # have to be > 0

            L_full = torch.linalg.cholesky(OU_covariance_full)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            assert inverse_full.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch)
            # Highlight: B.49 Λ−1aa
            inverse_internal = inverse_full[internal_indexes]
            inverse_internal = inverse_internal[:, internal_indexes]  # [z_dim,n_test,n_test]
            #assert inverse_internal.shape == (self.z_dim, self.n_internal_batch, self.n_internal_batch)
            assert inverse_internal.shape == (self.n_internal_batch, self.n_internal_batch)
            # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
            # Highlight: µa
            OU_mean_internal = torch.zeros((self.n_internal_batch,))  # [n_internal,]
            # Highlight: Λab
            inverse_internal_leaves = inverse_full[internal_indexes]  # [z_dim,n_test,n_test+n_train]---> [z_dim,n_train,]
            inverse_internal_leaves = inverse_internal_leaves[:,~internal_indexes]  # [z_dim,n_test,n_train]
            assert inverse_internal_leaves.shape == (self.n_internal_batch, self.n_leaves)
            # Highlight: xb
            eps_z = map_estimates["eps_z"]
            L_train = L_full[~internal_indexes]
            L_train = L_train[:,~internal_indexes]
            xb = (L_train@eps_z).T #[zdim, ntrain]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            inverse_internal_bis = torch.linalg.inv(inverse_internal)#https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
            part1 = torch.matmul(inverse_internal_bis, inverse_internal_leaves)  # og was [z_dim,n_test,n_train]
            assert part1.shape == (self.n_internal_batch,self.n_leaves)
            part2 = xb - OU_mean_leaves[None, :]  # [z_dim,n_train]
            OU_mean = OU_mean_internal[None, :,None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]
            assert OU_mean.squeeze(-1).shape == (self.z_dim, self.n_internal_batch)

            latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), inverse_internal_bis).to_event(1).sample()
            #latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1), torch.cholesky_inverse(Inverse_internal) + 1e-6).to_event(1).sample()

            latent_space = latent_space.T
            assert latent_space.shape == (self.n_internal_batch, self.z_dim)
            return {"latent_space": latent_space,"covariance": OU_covariance_full, "internal_idx": internal_indexes}

class DRAUPNIRModel_classic(DRAUPNIRModelClass):
    """Implements the ordinary version of Draupnir as described in the paper. It receives as an input the entire leaves dataset,
     uses a GRU as the mapping function and blosum embeddings"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim + self.aa_probs
        self.num_layers = 1
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim,
                                         self.input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum=None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        #batch_nodes = datasets["int"][:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-1, device=self.device):

            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim] #This can maybe be done with new axis solely
            blosum = self.blosum_weighted.repeat(latent_space.shape[0], 1).reshape(latent_space.shape[0],
                                                                                   self.align_seq_len,
                                                                                   self.aa_probs)  # [n_nodes,max_seq,21]
            blosum = self.embed(blosum)
            latent_space = torch.cat((latent_space, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional

            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):
                    logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                    pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]

        self.covariance = out_dict["covariance"]

    def model_delta_map(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum=None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]

        #batch_nodes = datasets["int"][:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-1, device=self.device):

            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim] #This can maybe be done with new axis solely
            blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
            blosum = self.embed(blosum)
            latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                                       self.gru_hidden_dim).contiguous()  # bidirectional

            #with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1), pyro.plate("plate_seq",aminoacid_sequences.shape[0],dim=-2):
            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):
                    logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                    pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]

            self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):
        if self.args.select_guide == "delta_map":
            self.model_delta_map(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)
        else:
            self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,patristic_matrix_eval,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        if use_test2: #MAP estimate
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            out_prediction_dict = self.conditional_samplingMAP(map_estimates,patristic_matrix)
            latent_space, covariance, internal_idx = out_prediction_dict["latent_space"], out_prediction_dict[ "covariance"], out_prediction_dict["internal_idx"]
            n_nodes = self.n_internal  # I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
            covariance = covariance[:, internal_idx]
            covariance = covariance[:, :, internal_idx]

        elif use_test:# Marginal posterior
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            out_prediction_dict = self.conditional_sampling(map_estimates,patristic_matrix)
            latent_space, covariance, internal_idx = out_prediction_dict["latent_space"], out_prediction_dict["covariance"], out_prediction_dict["internal_idx"]
            n_nodes = self.n_internal  # I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
            covariance = covariance[:, internal_idx]
            covariance = covariance[:, :, internal_idx]
        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves
            covariance = self.covariance

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_.shape[0], 1).reshape(latent_space_.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_ = torch.cat((latent_space_, blosum), dim=2)  # [n_nodes,align_seq_len,z_dim + 21]

        #with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2,subsample_size=n_nodes):
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
                                      kappa_psi=None,
                                      covariance=covariance
                                      )

        return sampling_out

class DRAUPNIRModel_classic_no_blosum(DRAUPNIRModelClass):
    """Implements the ordinary version of Draupnir without blosum embeddings.
    It receives as an input the entire leaf dataset, uses a GRU as the mapping function WITHOUT blosum embeddings"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum = None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        nodes_idx = datasets["int"][:, 0, 1]
        patristic_nodes = patristic_matrix_sorted[1:,0]
        assert torch.equal(nodes_idx,patristic_nodes), "Patristic matrix is disordered or the dataset, node indices must coincide"
        # Highlight: Register GRU module
        pyro.module("decoder", self.decoder)


        #with pyro.plate("plate_batch", dim=-1, device=self.device): #separated plate
        # Highlight: GP prior over the latent space
        out_dict = self.gp_prior(patristic_matrix_sorted)
        latent_space = out_dict["latent_space"]
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim]
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1), pyro.plate("plate_seq",aminoacid_sequences.shape[0],dim=-2):
                    logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                    pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]

        self.covariance = out_dict["covariance"]

    def model_delta_map(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum = None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
        # Highlight: Register GRU module
        pyro.module("decoder", self.decoder)
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional

            #with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1), pyro.plate("plate_seq",aminoacid_sequences.shape[0], dim=-2):
            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):
                    logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                    pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]

            self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):
        if self.args.select_guide == "delta_map":
            self.model_delta_map(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)
        else:
            self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,patristic_matrix_eval,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """NOTE TO SELF: in the other models the cladistic matrix becomes the patristic test matrix"""

        if use_test or use_test2:
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            out_prediction_dict = self.conditional_sampling(map_estimates,patristic_matrix)
            latent_space,covariance, internal_idx = out_prediction_dict["latent_space"],out_prediction_dict["covariance"], out_prediction_dict["internal_idx"]
            n_nodes = self.n_internal #I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
            covariance = covariance[:,internal_idx]
            covariance = covariance[:,:,internal_idx]

        else:
            latent_space = map_estimates["latent_z"].T
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves
            covariance = self.covariance

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)

        #with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2,subsample_size=n_nodes):
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
                                      kappa_psi=None,
                                      covariance=covariance)

        return sampling_out

class DRAUPNIRModel_classic_no_blosum_1nbA(DRAUPNIRModelClass): #not batching + experimental prior
    """Implements the ordinary version of Draupnir without blosum embeddings.
    It receives as an input the entire leaf dataset, uses a GRU as the mapping function WITHOUT blosum embeddings"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.gp_prior = self.gp_prior_experiment5 #if we need to do a lot of test, do the dictionary trick again

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum = None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        nodes_idx = datasets["int"][:, 0, 1]
        patristic_nodes = patristic_matrix_sorted[1:,0]
        assert torch.equal(nodes_idx,patristic_nodes), "Patristic matrix is disordered or the dataset, node indices must coincide"
        # Highlight: Register GRU module
        pyro.module("decoder", self.decoder)
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.plate("plate_len",  dim=-2):
                        logits = self.decoder.forward(
                            input=latent_space,
                            hidden=decoder_hidden)
                        pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]
        self.covariance = out_dict["covariance"]

    def model_delta_map(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum = None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
        # Highlight: Register GRU module
        pyro.module("decoder", self.decoder)

        #with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional

            #with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1), pyro.plate("plate_seq",aminoacid_sequences.shape[0], dim=-2):
            with pyro.plate("plate_len",aminoacid_sequences.shape[1], dim=-2):
                    logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                    pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences)  # aa_seq = [n_nodes,align_seq_len]

        self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):
        if self.args.select_guide == "delta_map":
            self.model_delta_map(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)
        else:
            self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,patristic_matrix_eval,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """NOTE TO SELF: in the other models the cladistic matrix becomes the patristic test matrix"""

        if use_test or use_test2:
            assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
            out_prediction_dict = self.conditional_sampling_experiment5(map_estimates,patristic_matrix)
            latent_space,covariance, internal_idx = out_prediction_dict["latent_space"],out_prediction_dict["covariance"], out_prediction_dict["internal_idx"]
            n_nodes = self.n_internal #I had to split it up because of some weird data cases (coral), otherwise family_data_test.shape[0] would have sufficed
            covariance = covariance[internal_idx]
            covariance = covariance[:,internal_idx]

        else:
            lambd = torch.exp(map_estimates["log_lambd"])
            patristic_matrix_eval = patristic_matrix_eval[1:,1:]
            covariance = OUKernel_Fast_experiment(None, lambd, None).forward(patristic_matrix_eval)
            L = torch.linalg.cholesky(covariance)
            latent_space = L @ map_estimates["eps_z"]
            assert latent_space.shape == (self.n_leaves, self.z_dim)
            n_nodes = self.n_leaves

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)

        #with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2,subsample_size=n_nodes):
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
                                      kappa_psi=None,
                                      covariance=covariance)

        return sampling_out#experimen

class DRAUPNIRModel_classic_batching(DRAUPNIRModelClass):
    """Implements independent batching. Selects n sequences (in tree level order or random) and generates independent Gaussian processes.
    It uses batched Blosum weighted average embeddings."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim + self.aa_probs
        self.decoder = RNNDecoder_Tiling(align_seq_len=self.align_seq_len,
                                            aa_probs=self.aa_probs,
                                            gru_hidden_dim=self.gru_hidden_dim,
                                            z_dim = self.z_dim,
                                            input_size=self.input_size,
                                            kappa_addition = self.kappa_addition,
                                            num_layers = self.num_layers,
                                            pretrained_params = self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
        self.internal_nodes_batch = None
        self.n_leaves_internal_batch = None

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):

        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
        batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        self.n_leaves_batch = aminoacid_sequences.shape[0]  # need this for sampling from a pretrained model

        with pyro.plate("plate_batch", dim=-1, device=self.device):

            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]

            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,
                                                                              self.z_dim)  # [n_nodes,max_seq,z_dim]
            blosum = self.blosum_weighted.repeat(latent_space.shape[0], 1).reshape(latent_space.shape[0],
                                                                                   self.align_seq_len,
                                                                                   self.aa_probs)  # [n_nodes,max_seq,21] #Highlight: it workedwith the entire blosum weighted matrix
            # blosum = batch_blosum.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.max_seq_len,self.aa_prob) #[n_nodes,max_seq,21] #only use the weighted average of the batch sequences
            blosum = self.embed(blosum)
            latent_space = torch.cat((latent_space, blosum), dim=2)  # [n_nodes,max_seq_len,z_dim + 21]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                                   self.gru_hidden_dim).contiguous()  # bidirectional

            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):
                logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

            self.n_leaves_batch = self.batch_size  # need this for sampling from a pretrained model
            self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):

        self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample_batched(self, map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling for large data sets"""

        out_prediction_dict = self.prediction_batching_preprocessing(map_estimates, patristic_matrix_full,
                                                                     patristic_matrix_test, batch_idx, use_test,
                                                                     use_test2)
        latent_space, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_.shape[0], 1).reshape(latent_space_.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_ = torch.cat((latent_space_, blosum), dim=2)  # [n_nodes,max_seq_len,z_dim + 21]

        with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2):
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
                                      kappa_psi=None,
                                      covariance=covariance)

        return sampling_out

class DRAUPNIRModel_classic_batching_no_blosum(DRAUPNIRModelClass):
    """Implements independent batching. Selects n sequences (in tree level order or random) and generates independent Gaussian processes.
    It uses batched Blosum weighted average embeddings."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim #+ self.aa_probs
        self.decoder = RNNDecoder_Tiling(align_seq_len=self.align_seq_len,
                                            aa_probs=self.aa_probs,
                                            gru_hidden_dim=self.gru_hidden_dim,
                                            z_dim = self.z_dim,
                                            input_size=self.input_size,
                                            kappa_addition = self.kappa_addition,
                                            num_layers = self.num_layers,
                                            pretrained_params = self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
        self.internal_nodes_batch = None
        self.n_leaves_internal_batch = None

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):

        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]

        self.n_leaves_batch = aminoacid_sequences.shape[0] #need this for sampling from a pretrained model
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-1, device=self.device):

            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim]

            # blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21] #Highlight: it workedwith the entire blosum weighted matrix
            ##blosum = batch_blosum.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.max_seq_len,self.aa_prob) #[n_nodes,max_seq,21] #only use the weighted average of the batch sequences
            # blosum = self.embed(blosum)
            # latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,max_seq_len,z_dim + 21]

            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional

            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):

                logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

            self.n_leaves_batch = self.batch_size  # need this for sampling from a pretrained model
            self.covariance = out_dict["covariance"]
            #print("inside model covariance", out_dict["covariance"].shape)

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):

         self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample_batched(self, map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling for large data sets"""


        out_prediction_dict = self.prediction_batching_preprocessing(map_estimates, patristic_matrix_full, patristic_matrix_test, batch_idx,use_test, use_test2)
        latent_space, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)

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
                                      kappa_psi=None,
                                      covariance=covariance)

        return sampling_out

class DRAUPNIRModel_classic_batching_no_blosum_1bA(DRAUPNIRModelClass): #embeddings experiments
    """Implements independent batching. Selects n sequences (in tree level order or random) and generates independent Gaussian processes.
    It uses batched Blosum weighted average embeddings."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim
        self.gru_hidden_dim = self.z_dim
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.decoder = RNNDecoder_CrossAttention(align_seq_len=self.align_seq_len,
                                            aa_probs=self.aa_probs,
                                            gru_hidden_dim=self.gru_hidden_dim,
                                            z_dim = self.z_dim,
                                            input_size=self.input_size,
                                            kappa_addition = self.kappa_addition,
                                            num_layers = self.num_layers,
                                            pretrained_params = self.pretrained_params)
        #self.embed = EmbedComplex_1b(self.aa_probs,self.embedding_dim, self.pretrained_params)
        self.internal_nodes_batch = None
        self.n_leaves_internal_batch = None

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):


        aminoacid_sequences = datasets["int"][:, 2:, 0]
        max_len = aminoacid_sequences.shape[1]
        batch_nodes = datasets["int"][:, 0, 1]

        self.n_leaves_batch = aminoacid_sequences.shape[0] #need this for sampling from a pretrained model
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module

        pyro.module("decoder", self.decoder)

        #with pyro.poutine.scale(scale=map_estimates["annealing_factor"] if map_estimates is not None else torch.Tensor([1.])):
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space_2d = out_dict["latent_space"]
            latent_space_3d = latent_space_2d.repeat(1, self.align_seq_len).reshape(latent_space_2d.shape[0],self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim]
            #todo: re-add blosum embedding here
            #positional_embeddings = self.pos_emb(torch.arange(self.align_seq_len,device=latent_space_2d.device)[None,:]).repeat(latent_space_2d.shape[0],1,1)
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space_2d.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
            if map_estimates is not None:
                encoder_hidden_states = map_estimates["rnn_hidden_states"]
            else:
                # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
                encoder_hidden_states = latent_space_3d


            with pyro.plate("plate_len", max_len, dim=-2):

                logits = self.decoder.forward(
                        input=latent_space_3d,
                        hidden=decoder_hidden,
                        encoder_hidden_states = encoder_hidden_states
                )


                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

                self.n_leaves_batch = self.batch_size  # need this for sampling from a pretrained model

        self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):

        self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample_batched_autoregressive(self, map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling for large data sets"""

        out_prediction_dict = self.prediction_batching_preprocessing(map_estimates, patristic_matrix_full, patristic_matrix_test, batch_idx,use_test, use_test2)
        latent_space_2d, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]
        # Highlight: GP prior over the latent space
        latent_space_3d = latent_space_2d.repeat(1, self.align_seq_len).reshape(latent_space_2d.shape[0],self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim]
        #todo: re-add blosum embedding here
        #positional_embeddings = self.pos_emb(torch.arange(self.align_seq_len,device=latent_space_2d.device)[None,:]).repeat(latent_space_2d.shape[0],1,1)
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space_2d.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        if map_estimates is not None:
            encoder_hidden_states = map_estimates["test"]["rnn_hidden_states"] if use_test or use_test2 else map_estimates["rnn_hidden_states"]  # [N,L,zdim]
            encoder_hidden_states = encoder_hidden_states[int(batch_idx[0]):] if batch_idx[1] is None else encoder_hidden_states[int(batch_idx[0]):int(batch_idx[1])]
        else:
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            encoder_hidden_states = latent_space_3d

        input_token = torch.ones(latent_space_3d.shape[0],1,latent_space_3d.shape[2])

        aa_sequences = []
        logits = []
        for idx in range(self.align_seq_len + 1):
            logit = self.decoder.forward(
                input=input_token.to(decoder_hidden.dtype),
                hidden=decoder_hidden,
                encoder_hidden_states=encoder_hidden_states
            )
            if use_argmax:#Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
                aa_token = torch.argmax(logit,dim=2)
            else:
                aa_token_samples = dist.Categorical(logits=logit).sample([n_samples])
                aa_token = torch.from_numpy(stats.mode((aa_token_samples.permute(1,0,2).detach().cpu()),axis=1).mode).to(input_token.device) #we take the most lilely amino acid
            input_token = (aa_token + latent_space_2d).unsqueeze(1) #todo: not sure about this
            if idx != 0: #skip the dummy token
                aa_sequences.append(aa_token if use_argmax else aa_token_samples)
                logits.append(logit)


        aa_sequences = torch.concat(aa_sequences,axis=1).unsqueeze(0) if use_argmax else torch.concat(aa_sequences,axis=2)#I add one dimension at the beginning to resemble 1 sample and not have to change all the plotting code
        logits = torch.concat(logits,axis=1)
        sampling_out = SamplingOutput(aa_sequences=aa_sequences.detach(),
                                      latent_space=latent_space_2d.detach(),
                                      logits=logits.detach(),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None,
                                      covariance=covariance)

        return sampling_out

    def sample_batched(self, map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling for large data sets"""

        out_prediction_dict = self.prediction_batching_preprocessing(map_estimates, patristic_matrix_full, patristic_matrix_test, batch_idx,use_test, use_test2)
        latent_space_2d, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]

        # Highlight: GP prior over the latent space
        latent_space_3d = latent_space_2d.repeat(1, self.align_seq_len).reshape(latent_space_2d.shape[0],self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim]
        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space_2d.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional

        if map_estimates is not None:
            encoder_hidden_states = map_estimates["test"]["rnn_hidden_states"] if use_test or use_test2 else map_estimates["rnn_hidden_states"]  # [N,L,zdim]
            encoder_hidden_states = encoder_hidden_states[int(batch_idx[0]):] if batch_idx[1] is None else encoder_hidden_states[int(batch_idx[0]):int(batch_idx[1])]
        else:
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            encoder_hidden_states = latent_space_3d

        logits = self.decoder.forward(
            input=latent_space_3d,
            hidden=decoder_hidden,
            encoder_hidden_states=encoder_hidden_states
        )
        if use_argmax:#Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
            aa_sequences = torch.argmax(logits,dim=2).unsqueeze(0)
        else:
            aa_sequences = dist.Categorical(logits=logits).sample([n_samples])

        sampling_out = SamplingOutput(aa_sequences=aa_sequences.detach(),
                                      latent_space=latent_space_2d.detach(),
                                      logits=logits.detach(),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None,
                                     covariance=covariance)

        return sampling_out

class DRAUPNIRModel_classic_batching_no_blosum_1bB(DRAUPNIRModelClass): #prior experiments
    """Implements independent batching. Selects n sequences (in tree level order or random) and generates independent Gaussian processes.
    It uses batched Blosum weighted average embeddings."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim #+ self.aa_probs
        self.decoder = RNNDecoder_Tiling(align_seq_len=self.align_seq_len,
                                            aa_probs=self.aa_probs,
                                            gru_hidden_dim=self.gru_hidden_dim,
                                            z_dim = self.z_dim,
                                            input_size=self.input_size,
                                            kappa_addition = self.kappa_addition,
                                            num_layers = self.num_layers,
                                            pretrained_params = self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
        self.internal_nodes_batch = None
        self.n_leaves_internal_batch = None

        self.gp_prior_batched = self.gp_priors_experiments_dict[self.args.prior_experiment]
        self.conditional_sampling_batch= self.conditional_sampling_batch_dict[self.args.prior_experiment]

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):


        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]

        self.n_leaves_batch = aminoacid_sequences.shape[0] #need this for sampling from a pretrained model
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-1, device=self.device):

            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim]

            # blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21] #Highlight: it workedwith the entire blosum weighted matrix
            # blosum = batch_blosum.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.max_seq_len,self.aa_prob) #[n_nodes,max_seq,21] #only use the weighted average of the batch sequences
            # blosum = self.embed(blosum)
            # latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,max_seq_len,z_dim + 21]

            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional

            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):

                logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

            self.n_leaves_batch = self.batch_size  # need this for sampling from a pretrained model
            self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):

        self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample_batched(self, map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling for large data sets"""


        out_prediction_dict = self.prediction_batching_preprocessing_experiment(map_estimates, patristic_matrix_full, patristic_matrix_test, batch_idx,use_test, use_test2)
        latent_space, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)

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
                                      kappa_psi=None,
                                      covariance= covariance
                                      )

        return sampling_out

class DRAUPNIRModel_xlstm_batching_no_blosum(DRAUPNIRModelClass):
    """Implements independent batching. Selects n sequences (in tree level order or random) and generates independent Gaussian processes.
    It uses batched Blosum weighted average embeddings."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim #+ self.aa_probs
        # self.decoder = RNNDecoder_Tiling(align_seq_len=self.align_seq_len,
        #                                  aa_probs=self.aa_probs,
        #                                  gru_hidden_dim=self.gru_hidden_dim,
        #                                  z_dim=self.z_dim,
        #                                  input_size=self.input_size,
        #                                  kappa_addition=self.kappa_addition,
        #                                  num_layers=self.num_layers,
        #                                  pretrained_params=self.pretrained_params)
        self.decoder = xLSTMDecoder(max_len=self.align_seq_len,
                                  input_size=self.z_dim,
                                  z_dim = self.z_dim,
                                  output_size = self.aa_probs)

        self.internal_nodes_batch = None
        self.n_leaves_internal_batch = None
    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):

        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
        self.n_leaves_batch = aminoacid_sequences.shape[0] #need this for sampling from a pretrained model
        batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,
                                                                              self.z_dim)  # [n_nodes,max_seq,z_dim]

            if map_estimates is not None:
                embeddings = map_estimates["embeddings"]
                latent_space = embeddings + latent_space  # independent vectors

            #decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):

                logits = self.decoder.forward(
                        input=latent_space,
                        #hidden=decoder_hidden
                )
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

            self.n_leaves_batch = self.batch_size  # need this for sampling from a pretrained model
            self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):

        self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample_batched(self, map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling for large data sets"""

        out_prediction_dict = self.prediction_batching_preprocessing(map_estimates, patristic_matrix_full, patristic_matrix_test, batch_idx,use_test, use_test2)
        latent_space, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]

        #decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        if map_estimates is not  None:
            embeddings = map_estimates["test"]["embeddings"] if use_test or use_test2 else  map_estimates["embeddings"]
            embeddings = embeddings[int(batch_idx[0]):] if batch_idx[1] is None else embeddings[int(batch_idx[0]):int(batch_idx[1])]
            latent_space_ = embeddings + latent_space_

        logits = self.decoder.forward(
            input=latent_space_)

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
                                      kappa_psi=None,
                                      covariance=covariance
                                      )

        return sampling_out

class DRAUPNIRModel_miniRNN_batching_no_blosum(DRAUPNIRModelClass):
    """Implements independent batching. Selects n sequences (in tree level order or random) and generates independent Gaussian processes.
    It uses batched Blosum weighted average embeddings."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim #+ self.aa_probs
        self.decoder = miniGRUDecoder(depth=3,
                                      input_dim=self.z_dim,
                                      output_dim=self.aa_probs)
        self.internal_nodes_batch = None
        self.n_leaves_internal_batch = None

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):

        pyro.module("decoder", self.decoder)
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]

        self.n_leaves_batch = aminoacid_sequences.shape[0] #need this for sampling from a pretrained model
        batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)

        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1, self.align_seq_len).reshape(latent_space.shape[0], self.align_seq_len,self.z_dim)  # [n_nodes,max_seq,z_dim]
            if map_estimates is not None:
                embeddings = map_estimates["embeddings"]
                latent_space = embeddings + latent_space  # independent vectors

            #decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):

                logits = self.decoder.forward(
                        x=latent_space,
                        #prev_hiddens=embeddings
                )
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

            self.n_leaves_batch = self.batch_size  # need this for sampling from a pretrained model
            self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):

        self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample_batched(self, map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling for large data sets"""


        out_prediction_dict = self.prediction_batching_preprocessing(map_estimates, patristic_matrix_full, patristic_matrix_test, batch_idx,use_test, use_test2)
        latent_space, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]

        #decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        if map_estimates is not None:
            embeddings = map_estimates["test"]["embeddings"] if use_test or use_test2 else  map_estimates["embeddings"]
            embeddings = embeddings[int(batch_idx[0]):] if batch_idx[1] is None else embeddings[int(batch_idx[0]):int(batch_idx[1])]
            latent_space_ = embeddings + latent_space_

        logits = self.decoder.forward(x=latent_space_)

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
                                      kappa_psi=None,
                                      covariance=covariance
                                      )

        return sampling_out

class DRAUPNIRModel_transformer_batching_no_blosum(DRAUPNIRModelClass):
    """Implements independent batching. Selects n sequences (in tree level order or random) and generates independent Gaussian processes.
    It uses batched Blosum weighted average embeddings."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        #self.input_size = self.z_dim
        self.decoder = TransformerDecoder(
                                            input_dim_l= self.z_dim,
                                            input_dim_r= self.z_dim,
                                          align_seq_len = self.align_seq_len + 1,
                                          hidden_dim = self.gru_hidden_dim,
                                          output_dim = self.aa_probs)
        #self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
        self.internal_nodes_batch = None
        self.n_leaves_internal_batch = None
        self.bos_embedding = nn.Parameter(torch.randn(self.z_dim)) # start token , needed for sequence generation

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):

        aminoacid_sequences = datasets["int"][:, 2:, 0]
        sequences_blosum = datasets["blosum"]
        bos_token = self.bos_embedding[None,None,:].expand(sequences_blosum.shape[0], -1, -1 )
        #sequences_blosum = torch.concat((bos_token[:,:,:self.aa_probs],sequences_blosum), dim=1) # add the start token
        batch_nodes = datasets["int"][:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        #pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"] # the latent space has been generated from the cls token
            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]

            if map_estimates is not None:
                context_vector, hidden_states = map_estimates["context_vector"], map_estimates["hidden_states"] # hidden states will become the esm2 embeddings eventually
                latent_space = torch.concat((bos_token, latent_space), dim=1)  # add the start token
                hidden_states = torch.concat((bos_token, hidden_states), dim=1)
            else:
                latent_space = torch.concat((bos_token, latent_space), dim=1)  # add the start token
                hidden_states = latent_space

            # blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21] #Highlight: it workedwith the entire blosum weighted matrix
            # #blosum = batch_blosum.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.max_seq_len,self.aa_prob) #[n_nodes,max_seq,21] #only use the weighted average of the batch sequences
            # blosum = self.embed(blosum)
            # latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,max_seq_len,z_dim + 21]

            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):

                #todo: if using sequences_blosum directly is too easy, then think about some distorsion
                logits = self.decoder.forward(hidden_states, latent_space)

                logits = logits[:,1:] # remove the start token
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

            self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):

        self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample_batched(self, map_estimates, n_samples, datasets_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling for large data sets"""

        if use_test or use_test2:# Only Marginal posterior available when batching
            assert patristic_matrix_full[1:,1:].shape == (self.n_all,self.n_all)
            #Highlight: Slice out the train sequences and only a batch from the test sequences
            if batch_idx[1] is None:
                self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0]) + 1:, 0]
            else:
                self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0])+1:int(batch_idx[1])+1,0]
            self.n_internal_batch = len(self.internal_nodes_batch)
            nodes_batch = torch.cat((self.leaves_nodes,self.internal_nodes_batch))
            self.n_leaves_internal_batch = len(nodes_batch)
            indexes = (patristic_matrix_full[:, 0][..., None] == nodes_batch).any(-1)
            indexes[0] = True #re-add the nodes names
            patristic_matrix = patristic_matrix_full[indexes]
            patristic_matrix = patristic_matrix[:,indexes]
            out_prediction_dict = self.conditional_sampling_batch(map_estimates,patristic_matrix)
            latent_space = out_prediction_dict["latent_space"]
            covariance = out_prediction_dict["covariance"]
            n_nodes = self.n_internal_batch
            hidden_states_key = "hidden_states_test"

        else: #training/leaves
            n_nodes = self.n_leaves_batch #here n_leaves has been overloaded by the batch size
            latent_space = map_estimates["latent_z"].T
            latent_space = latent_space[int(batch_idx[0]):int(batch_idx[1])]
            hidden_states_key = "hidden_states"
            covariance = self.covariance[batch_idx[0]:batch_idx[1]] if batch_idx[1] is None else self.covariance["covariance"][batch_idx[0]:] #this should be in the same order  as the map estimates
            assert latent_space.shape == (n_nodes, self.z_dim)

        bos_token = self.bos_embedding[None,None,:].expand(latent_space.shape[0], -1, -1 )
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)


        # blosum = self.blosum_weighted.repeat(latent_space_.shape[0], 1).reshape(latent_space_.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        # blosum = self.embed(blosum)
        # latent_space_ = torch.cat((latent_space_, blosum), dim=2)  # [n_nodes,max_seq_len,z_dim + 21]

        if map_estimates is not None:
            context_vector, hidden_states = map_estimates["context_vector"], map_estimates[hidden_states_key] # hidden states will become the esm2 embeddings eventually
            if batch_idx[1] is None:
                hidden_states = hidden_states[int(batch_idx[0]):]
            else:
                hidden_states = hidden_states[int(batch_idx[0]):int(batch_idx[1])] # we need to slice the batch as well here

            latent_space_ = torch.concat((bos_token, latent_space_), dim=1)  # add the start token
            hidden_states = torch.concat((bos_token, hidden_states), dim=1)
        else:
            latent_space_ = torch.concat((bos_token, latent_space_), dim=1)  # add the start token
            hidden_states = latent_space_

        with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2):

            logits = self.decoder.forward(hidden_states, latent_space_) # i will not have the test sequences, therefore they should not be anywhere near the decoder
            logits = logits[:,1:]

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
                                      kappa_psi=None,
                                      covariance=covariance
                                      )

        return sampling_out


#todo: delete from down here

class DRAUPNIRModel_cladebatching(DRAUPNIRModelClass):
    """Perform inference by dividing the tree into batches that correspond to the clade in the tree, with its corresponent batched latent space.
    Clade batch training and sampling."""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim + self.aa_probs
        #self.decoder_attention = RNNAttentionDecoder(self.n_leaves, self.max_seq_len, self.aa_prob, self.gru_hidden_dim,self.input_size,self.embedding_dim, self.z_dim, self.kappa_addition)
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)

    def model_delta_map(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        #batch_nodes = datasets["int"][:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        self.n_leaves_batch = datasets["int"].shape[0]

        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space of all the leaves
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
            #blosum = batch_blosum.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.max_seq_len,self.aa_prob) #[n_nodes,max_seq,21]
            blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
            blosum = self.embed(blosum)
            batch_latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,max_seq_len,z_dim + 21]
            #batch_latent_space = latent_space[batch_indexes] #Highlight: For plating; In order to reduce the load on the GRU memory we split the latent space of the leaves by clades/batch
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                                   batch_latent_space.shape[0],
                                                   self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):
                logits = self.decoder.forward(
                        input=batch_latent_space,
                        hidden=decoder_hidden)
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

            self.covariance = out_dict["covariance"]

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        #batch_nodes = datasets["int"][:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        self.n_leaves_batch = datasets["int"].shape[0]
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space of all the leaves
            out_dict = self.gp_prior_batched(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
            #blosum = batch_blosum.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.max_seq_len,self.aa_prob) #[n_nodes,max_seq,21]
            blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
            blosum = self.embed(blosum)
            batch_latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,max_seq_len,z_dim + 21]
            #batch_latent_space = latent_space[batch_indexes] #Highlight: For plating; In order to reduce the load on the GRU memory we split the latent space of the leaves by clades/batch
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                                   batch_latent_space.shape[0],
                                                   self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.plate("plate_len", dim=-2):
                logits = self.decoder.forward(
                        input=batch_latent_space,
                        hidden=decoder_hidden)
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,max_seq_len]

        self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):
        if self.args.select_guide == "delta_map":
            self.model_delta_map(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)
        else:
            self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample(self,  map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Full latent space inference, plating sequences by clade"""
        # if use_test:
        #     assert patristic_matrix[1:,1:].shape == (self.n_all,self.n_all)
        #     latent_space = self.conditional_sampling(map_estimates,patristic_matrix)
        #     n_nodes = self.n_internal #DO NOT REMOVE: I had to write this line because of some weird data cases (coral dataset), otherwise family_data_test.shape[0] would have sufficed
        # elif use_test2:
        #     #latent_space = self.conditional_sampling_descendants(map_estimates,patristic_matrix)
        #     latent_space = self.conditional_sampling_descendants_leaves(map_estimates,patristic_matrix)
        #     n_nodes = self.n_internal
        # else:
        #     latent_space = map_estimates["latent_z"].T
        #     assert latent_space.shape == (self.n_leaves, self.z_dim)
        #     n_nodes = self.n_leaves

        out_prediction_dict = self.prediction_batching_preprocessing(map_estimates, patristic_matrix_full, patristic_matrix_test, batch_idx,use_test, use_test2)
        latent_space, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_extended = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_extended.shape[0], 1).reshape(latent_space_extended.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_extended = torch.cat((latent_space_extended, blosum), dim=2)  # [n_nodes,max_seq_len,z_dim + 21]

        #with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2):
        logits = self.decoder.forward(
                input=latent_space_extended,
                hidden=decoder_hidden)
        if use_argmax: #Pick the sequence with the highest likelihood, now n_samples, n_samples = 1
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

    def sample_batched(self,  map_estimates, n_samples, family_data_test, patristic_matrix_full,patristic_matrix_test,batch_idx=None,use_argmax=False,use_test=True,use_test2=False):
        """Batched sampling based on clade membership"""

        out_prediction_dict = self.prediction_batching_preprocessing(map_estimates, patristic_matrix_full, patristic_matrix_test, batch_idx,use_test, use_test2)
        latent_space, n_nodes, covariance = out_prediction_dict["latent_space"], out_prediction_dict["n_nodes"], out_prediction_dict["covariance"]

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],self.gru_hidden_dim).contiguous()  # Not bidirectional
        latent_space_ = latent_space.repeat(1, self.align_seq_len).reshape(n_nodes,self.align_seq_len, self.z_dim)
        blosum = self.blosum_weighted.repeat(latent_space_.shape[0], 1).reshape(latent_space_.shape[0], self.align_seq_len,self.aa_probs)  # [n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space_ = torch.cat((latent_space_, blosum), dim=2)  # [n_nodes,max_seq_len,z_dim + 21]


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
                                      kappa_psi=None,
                                      covariance=covariance)

        return sampling_out

class DRAUPNIRModel_leaftesting(DRAUPNIRModelClass):
    """Leaves training and testing. Train on full leave latent space (train + test), only observe the pre-selected train leaves"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim + self.aa_probs
        #self.decoder_attention = RNNAttentionDecoder(self.n_leaves, self.align_seq_len, self.aa_probs, self.gru_hidden_dim,self.input_size,self.embedding_dim, self.z_dim, self.kappa_addition)
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
    def model_delta_map(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum=None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        #angles = datasets["int"][:, 2:, 1:3]
        train_nodes = datasets["int"][:, 0, 1]
        train_indexes = (patristic_matrix_sorted[1:, 0][..., None] == train_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-1, device=self.device):

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
            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):
                logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                #Highlight: Observe only some of the leaves
                logits = logits[train_indexes]
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,align_seq_len]

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum=None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        #angles = datasets["int"][:, 2:, 1:3]
        train_nodes = datasets["int"][:, 0, 1]
        train_indexes = (patristic_matrix_sorted[1:, 0][..., None] == train_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        with pyro.plate("plate_batch", dim=-1, device=self.device):
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
            with pyro.plate("plate_len", dim=-2):
                logits = self.decoder.forward(
                        input=latent_space,
                        hidden=decoder_hidden)
                #Highlight: Observe only some of the leaves
                logits = logits[train_indexes]
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,align_seq_len]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):
        if self.args.select_guide == "delta_map":
            self.model_delta_map(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)
        else:
            self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,patristic_matrix_eval,use_argmax=False,use_test=True,use_test2=False):
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

        #with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2):
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
    """Leaves training and testing.Predicting both ANGLES and AA sequences. Working on full or partial leaves space depending on the
    leaves_testing argument stated in datasets.py"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim + self.aa_probs
        self.decoder = RNNDecoder_Tiling_Angles(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
    def model_delta_map(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum=None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        angles = datasets["int"][:, 2:, 1:3]
        angles_mask = torch.where(angles == 0., angles, 1.).type(angles.dtype) #keep as 0 the gaps and set to 1 where there is an observation
        train_nodes = datasets["int"][:, 0, 1]
        train_indexes = (patristic_matrix_sorted[1:, 0][..., None] == train_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)

        with pyro.plate("plate_batch", dim=-1, device=self.device):

            # Highlight: GP prior over the latent space of all the leaves
            out_dict = self.gp_prior(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention

            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]

            blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
            blosum = self.embed(blosum)
            latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                                   latent_space.shape[0],
                                                   self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-2):
                logits,means,kappas = self.decoder.forward(
                    input=latent_space,
                    hidden=decoder_hidden)
                logits = logits[train_indexes]
                means = means[train_indexes]
                kappas = kappas[train_indexes]
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,align_seq_len]
                pyro.sample("phi",dist.VonMises(loc = means[:,:,0],concentration = kappas[:,:,0]).mask(angles_mask), obs=angles[:,:,0])
                pyro.sample("psi",dist.VonMises(loc = means[:,:,1],concentration = kappas[:,:,1]).mask(angles_mask), obs=angles[:,:,1])

            self.covariance = out_dict["covariance"]

    def model_variational(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum=None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        angles = datasets["int"][:, 2:, 1:3]
        angles_mask = torch.where(angles == 0., angles, 1.).type(angles.dtype) #keep as 0 the gaps and set to 1 where there is an observation
        train_nodes = datasets["int"][:, 0, 1]
        train_indexes = (patristic_matrix_sorted[1:, 0][..., None] == train_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space of all the leaves
            out_dict = self.gp_prior(patristic_matrix_sorted)
            latent_space = out_dict["latent_space"]
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]

            blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
            blosum = self.embed(blosum)
            latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]
            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2,
                                                   latent_space.shape[0],
                                                   self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.plate("plate_len", dim=-2):
                logits,means,kappas = self.decoder.forward(
                    input=latent_space,
                    hidden=decoder_hidden)
                logits = logits[train_indexes]
                means = means[train_indexes]
                kappas = kappas[train_indexes]
                pyro.sample("aa_sequences", dist.Categorical(logits=logits), obs=aminoacid_sequences) #aa_seq = [n_nodes,align_seq_len]
                pyro.sample("phi",dist.VonMises(loc = means[:,:,0],concentration = kappas[:,:,0]).mask(angles_mask), obs=angles[:,:,0])
                pyro.sample("psi",dist.VonMises(loc = means[:,:,1],concentration = kappas[:,:,1]).mask(angles_mask), obs=angles[:,:,1])

        self.covariance = out_dict["covariance"]

    def model(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum,map_estimates):
        if self.args.select_guide == "delta_map":
            self.model_delta_map(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)
        else:
            self.model_variational(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,batch_blosum, map_estimates)

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,patristic_matrix_eval,use_argmax=False,use_test=True,use_test2=False):
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

        #with pyro.plate("plate_len",self.align_seq_len, dim=-1), pyro.plate("plate_seq",n_nodes,dim=-2):

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

class DRAUPNIRModel_classic_plating(DRAUPNIRModelClass):
    """Implements the plated version of Draupnir.
     a) It receives as an input the entire leaves dataset
     b) plates or subsamples the sequences when mapping them to the observations, no blosum embedding split
     c) uses a GRU as the mapping function.
    NOTE: The plating of the leaves nodes can be with the ordered nodes (same order as input) or random order"""
    def __init__(self,ModelLoad):
        DRAUPNIRModelClass.__init__(self,ModelLoad)
        self.input_size = self.z_dim + self.aa_probs
        self.num_layers = 2
        self.decoder = RNNDecoder_Tiling(self.align_seq_len, self.aa_probs, self.gru_hidden_dim, self.z_dim, self.input_size,self.kappa_addition,self.num_layers,self.pretrained_params)
        self.embed = EmbedComplex(self.aa_probs,self.embedding_dim, self.pretrained_params)
        self.splitted_leaves_indexes = list(torch.tensor_split(torch.arange(self.n_leaves), int(self.n_leaves / self.plate_size)) * self.num_epochs)
        if self.plate_unordered:
            self.model = self.model_unordered
        else:
            self.model = self.model_ordered
    def model_delta_map_ordered(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum = None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        # Highlight: GP prior over the latent space
        latent_space = self.gp_prior(patristic_matrix_sorted)
        # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
        latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
        blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
        blosum = self.embed(blosum)
        latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]

        decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                               self.gru_hidden_dim).contiguous()

        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2, subsample=self.splitted_leaves_indexes.pop(0)) as indx:  # Highlight: Ordered subsampling
                logits = self.decoder.forward(
                    input=latent_space[indx],
                    hidden=decoder_hidden[:,indx])
                pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences[indx])  # aa_seq = [n_nodes,align_seq_len]

    def model_variational_ordered(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum = None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
        #batch_indexes = (patristic_matrix_sorted[1:, 0][..., None] == batch_nodes).any(-1)
        # Highlight: Register GRU module
        pyro.module("embeddings",self.embed)
        pyro.module("decoder", self.decoder)
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            # Highlight: GP prior over the latent space
            latent_space = self.gp_prior(patristic_matrix_sorted)
            # Highlight: MAP the latent space to logits using the Decoder from a Seq2seq model with/without attention
            latent_space = latent_space.repeat(1,self.align_seq_len).reshape(latent_space.shape[0],self.align_seq_len,self.z_dim) #[n_nodes,max_seq,z_dim]
            blosum = self.blosum_weighted.repeat(latent_space.shape[0],1).reshape(latent_space.shape[0],self.align_seq_len,self.aa_probs) #[n_nodes,max_seq,21]
            blosum = self.embed(blosum)
            latent_space = torch.cat((latent_space,blosum),dim=2) #[n_nodes,align_seq_len,z_dim + 21]

            decoder_hidden = self.h_0_MODEL.expand(self.decoder.num_layers * 2, latent_space.shape[0],
                                                   self.gru_hidden_dim).contiguous()

            with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
                with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2, subsample=self.splitted_leaves_indexes.pop(0)) as indx:  # Highlight: Ordered subsampling
                    logits = self.decoder.forward(
                        input=latent_space[indx],
                        hidden=decoder_hidden[:,indx])
                    pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences[indx])  # aa_seq = [n_nodes,align_seq_len]

    def model_delta_map_unordered(self, datasets, patristic_matrix_sorted,patristic_matrix_eval,data_blosum,batch_blosum = None,map_estimates=None):
        aminoacid_sequences = datasets["int"][:, 2:, 0]
        batch_nodes = datasets["int"][:, 0, 1]
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
                                               self.gru_hidden_dim).contiguous()

        with pyro.plate("plate_len", aminoacid_sequences.shape[1], dim=-1):
            with pyro.plate("plate_seq",aminoacid_sequences.shape[0],dim=-2,subsample_size=self.plate_size) as indx:#Highlight: Random subsampling
            #with pyro.plate("plate_seq", aminoacid_sequences.shape[0], dim=-2,subsample=self.splitted_leaves_indexes.pop(0)) as indx:  # Highlight: Ordered subsampling
                logits = self.decoder.forward(
                    input=latent_space[indx],
                    hidden=decoder_hidden[:,indx])
                pyro.sample("aa_sequences", dist.Categorical(logits=logits),obs=aminoacid_sequences[indx])  # aa_seq = [n_nodes,align_seq_len]

    def model(self, datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum, batch_blosum,map_estimates):
        if self.args.select_guide == "delta_map":
            if self.args.plate_unordered:
                self.model_delta_map_unordered(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,
                                     batch_blosum, map_estimates)
            else:
                self.model_delta_map_ordered(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,
                                     batch_blosum, map_estimates)
        else:
            raise ValueError("Under construction")
            if self.args.plate_unordered:
                self.model_variationl_unordered(sdatasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,
                                     batch_blosum, map_estimates)
            else:
                self.model_variational_ordered(datasets, patristic_matrix_sorted, patristic_matrix_eval, data_blosum,
                                       batch_blosum, map_estimates)

    def sample(self, map_estimates, n_samples, family_data_test, patristic_matrix,patristic_matrix_eval,use_argmax=False,use_test=True,use_test2=False):
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
