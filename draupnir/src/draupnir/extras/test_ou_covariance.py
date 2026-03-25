import torch
from abc import ABC, abstractmethod
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
    def __init__(self, sigma_f, lambd, sigma_n = None):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lambd = lambd
    def preforward(self,t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Not used function but needs to be here"""

        return torch.zeros((1,))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        #cov_b = torch.exp(-distance_matrix / _lambd) * _sigma_f ** 2 + _sigma_n + torch.eye(self.n_b*2, device=self.device) * 1e-5
        lambd = self.lambd.unsqueeze(-1).unsqueeze(-1) #self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)

        if self.sigma_f is not None:
            first_term = self.sigma_f ** 2
            first_term = first_term.unsqueeze(-1).unsqueeze(-1) #[:,None,None]

            noise = torch.eye(t.shape[0]) #distributes noise/stochascity to diagonal of the covariance
            if self.sigma_n is not None:
                sigma_n = self.sigma_n.unsqueeze(-1).unsqueeze(-1)
                return first_term * second_term + sigma_n ** 2 * noise
            else:

                return first_term * second_term + noise*1e-6
        else:
            return  second_term


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

        #DUMMY SAMPLING!!!!!!!!!!!!!!!!!!!!

        rho = dist.Beta(8,2).sample() # correlation prior
        rho = rho.clamp(1e-8,1-1e-8)
        lambd = -2/torch.log(rho)

        patristic_matrix = patristic_matrix_sorted[1:, 1:] #[n_leaves_batch,n_leaves_batch]
        OU_covariance = OUKernel_Fast(None, lambd, None).forward(patristic_matrix) #[z_dim, n_leaves,n_leaves ]
        #assert OU_covariance.shape == (self.z_dim, self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        assert OU_covariance.shape == (self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        #invertibility checkpoints:
        if torch.allclose(OU_covariance,OU_covariance.T, atol=1e-6):
            print("matrix is symmetric")
        else:
            print("matrix is not symmetric")
        eigvals = torch.linalg.eigvalsh(OU_covariance)
        print("min eigenvalue per batch",eigvals.min(dim=-1).values) #have to be > 0
        L = torch.linalg.cholesky(OU_covariance)
        eps_z = dist.Normal(0,1).sample([self.n_leaves_batch, self.z_dim]) #adds some noise to each of the leaves?
        latent_space = L @ eps_z
        assert latent_space.shape == (self.n_leaves_batch,self.z_dim)


        #latent_space = latent_space.T
        return {
                "eps_z": eps_z,
                "latent_z": latent_space,
                "rho": rho,
                "lambd": lambd
                }

    def conditional_sampling_batch(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""
            rho = map_estimates["rho"]
            rho = rho.clamp(1e-8,1-1e-8)
            lambd = -2 / torch.log(rho)

            internal_indexes = (patristic_matrix[1:, 0][..., None] == self.internal_nodes_batch).any(-1)
            #n_internal = family_data_test.shape[0]
            # Highlight: Sample the ancestors conditiones on the leaves (by using the full patristic matrix). See Page 689 at Patter Recongnition and Ml (Bishop)
            # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves
            patristic_matrix_batch = patristic_matrix[1:, 1:] #remove the node names
            assert patristic_matrix_batch.shape == (self.n_leaves_internal_batch, self.n_leaves_internal_batch), "Here we are using a slice of the patristic matrix with size n_leaves_batch = batch_size!"
            OU = OUKernel_Fast(None, lambd, None)
            OU_covariance_full = OU.forward(patristic_matrix_batch)
            L_full = torch.linalg.cholesky(OU_covariance_full)
            # Highlight: Calculate the inverse of the covariance matrix Λ ≡ Σ−1
            inverse_full = torch.linalg.inv(OU_covariance_full)  # [z_dim,n_test+n_train,n_test+n_train]
            #assert inverse_full.shape == (self.z_dim, self.n_leaves_internal_batch, self.n_leaves_internal_batch)
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
            #assert inverse_internal_leaves.shape == (self.z_dim, self.n_internal_batch, self.n_leaves)
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
            return latent_space


ultrametric = True
suffix = "_ultrametric" if ultrametric else ""

train_patristic = torch.load(f"data/train_info_dict{suffix}.torch",weights_only=False)["patristic"].detach().cpu() #only leaves
test_patristic = torch.load(f"data/test_info_dict{suffix}.torch",weights_only=False)["patristic"].detach().cpu() #only internal
full_patristic = torch.load(f"data/full_patristic_matrix{suffix}.torch",weights_only=False).detach().cpu() # leaves + internal


train_idx = (full_patristic[:, 0][..., None] == train_patristic[0,1:]).any(-1)
train_idx[0] = True #re-add node names
train_patristic = full_patristic[train_idx]
train_patristic = train_patristic[:,train_idx]

root_index = (full_patristic[:, 0][..., None] == torch.Tensor([0.])).any(-1)
root_index[0] = False  # remove nodes names
tree_height = full_patristic[root_index].max()
tree_diameter = full_patristic[1:,1:].max() #maximum pairwise distance

print("Tree Height",tree_height)
print("Tree diameter",tree_diameter)
print("Test diameter", test_patristic[1:,1:].max())


exit()

map_estimates = TestModel(test_patristic,tree_height).gp_prior_batched(train_patristic)

latent_test = TestModel(test_patristic,tree_height).conditional_sampling_batch(map_estimates,full_patristic)

print("out latent test",latent_test.shape)
print(latent_test)



