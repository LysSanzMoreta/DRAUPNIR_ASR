import torch
from abc import ABC, abstractmethod
import pyro
from pyro import distributions as dist
torch.set_default_dtype(torch.float64)
class GPKernel(ABC):
    @abstractmethod
    def preforward(self, t1: torch.Tensor,t2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
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
    def __init__(self, sigma_f, lamb, sigma_n = None):
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

        if self.sigma_n is not None:
            sigma_n = self.sigma_n.unsqueeze(-1).unsqueeze(-1)
            return first_term * second_term + sigma_n ** 2 * noise
        else:

            return first_term * second_term + noise*1e-6

class TestModel():

    def __init__(self,test_patristic, tree_height):

        self.z_dim = 30
        self.n_leaves = 200
        self.n_leaves_batch = 200 #hard coded
        self.n_leaves_internal_batch = 399
        self.n_internal_batch = 199
        self.internal_nodes_batch = test_patristic[0,1:]
        self.tree_height = tree_height

    def gp_prior_batched(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"
        # Highlight; OU kernel parameters
        # alpha = pyro.sample("alpha", dist.HalfNormal(1).expand_by([3, ]).to_event(0))
        # sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand_by([self.z_dim, ]).to_event(0))  # rate of mean reversion/selection strength---> signal variance #removed .to_event(1)...
        # sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand_by([self.z_dim, ]).to_event(0))  # Gaussian noise
        # lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand_by([self.z_dim, ]).to_event(0))  # characteristic length-scale
        #
        #
        #DUMMY SAMPLING!!!!!!!!!!!!!!!!!!!!

        sigma_f = dist.HalfNormal(1).sample([self.z_dim]) + 1e-6
        log_lambd = dist.Normal(torch.log(self.tree_height/2),0.5).sample([self.z_dim]) #assuming neperian logarithm
        lambd = torch.exp(log_lambd)

        # Highlight: Sample the latent space from MultivariateNormal with GP prior on covariance
        patristic_matrix = patristic_matrix_sorted[1:, 1:]
        OU_covariance = OUKernel_Fast(sigma_f, lambd, None).forward(patristic_matrix)



        OU_mean = torch.zeros((patristic_matrix.shape[0],)).unsqueeze(0)

        assert OU_covariance.shape == (self.z_dim, self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        assert OU_mean.shape == (1,self.n_leaves_batch)
        #noise = 1e-15 + torch.eye(OU_covariance.shape[1])
        #https://github.com/pyro-ppl/pyro/issues/702
        #https://forum.pyro.ai/t/runtimeerror-during-cholesky-decomposition/1216/2---> fix runtime error with choleky decomposition
        #https://forum.pyro.ai/t/using-constraints-within-an-nn-module/486
        #OU_covariance = transform_to(constraints.lower_cholesky)(OU_covariance) #check that this does not affect performance
        latent_space = pyro.sample('latent_z', dist.MultivariateNormal(OU_mean, OU_covariance ).to_event(1)) #[z_dim=30,n_nodes] #+ noise[None,:,:]
        #latent_space = latent_space.T
        return {"latent_z": latent_space,
                "alpha": None,
                "sigma_f": sigma_f,
                "lambd": lambd


                }

    def conditional_sampling_batch(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""
            sigma_f = map_estimates["sigma_f"] + 1e-6
            lambd = map_estimates["lambd"] + 1e-6

            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes_batch).any(-1)
            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_batch = patristic_matrix[1:, 1:] #remove the node names
            assert patristic_matrix_batch.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch), "Here we are using a slice of the patristic matrix with size n_leaves_batch = batch_size!"
            OU = OUKernel_Fast(sigma_f, lambd, None)
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
            xb = map_estimates["latent_z"]  # [z_dim,n_train]
            # Highlight:µb
            OU_mean_leaves = torch.zeros((self.n_leaves,))
            # Highlight:µa|b---> Splitted Equation  B-50
            inverse_internal_bis = torch.linalg.inv(inverse_internal)#https://stackoverflow.com/questions/79417996/efficient-matrix-inversion-multiplication-with-multiple-batch-dimensions-in-pyto
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
            return latent_space



from sklearn.preprocessing import MinMaxScaler, minmax_scale

import sklearn
scaler = MinMaxScaler()


train_patristic = torch.load("data/train_info_dict.torch",weights_only=False)["patristic"].detach().cpu() #only leaves
test_patristic = torch.load("data/test_info_dict.torch",weights_only=False)["patristic"].detach().cpu() #only internal
full_patristic = torch.load("data/full_patristic_matrix.torch",weights_only=False).detach().cpu() # leaves + internal


print(full_patristic[1:,1:].max())




full_patristic[1:,1:] = torch.from_numpy(sklearn.preprocessing.minmax_scale(full_patristic[1:,1:],feature_range=(0,2),axis=0))



train_idx = (full_patristic[:, 0][..., None] == train_patristic[0,1:]).any(-1)



train_patristic = full_patristic[train_idx]
train_patristic = train_patristic[:,train_idx]


print(train_patristic[1:,1:].max())
print(full_patristic[1:,1:].max())



root_index = (full_patristic[:, 0][..., None] == torch.Tensor([0.])).any(-1)
root_index[0] = False  # remove nodes names
tree_height = full_patristic[root_index].max()

tree_diameter = full_patristic[1:,1:].max() #maximum pairwise distance



print(tree_height)

print(tree_diameter)

exit()

# offdiag_zeros = (full_patristic == 0).logical_and(~torch.eye(full_patristic.size(0), dtype=torch.bool)).sum()
# print(offdiag_zeros)

#TODO: batched version, and version without sigma_n
map_estimates = TestModel(test_patristic,tree_height).gp_prior_batched(train_patristic)
latent_test = TestModel(test_patristic,tree_height).conditional_sampling_batch(map_estimates,full_patristic)




print(latent_test)



