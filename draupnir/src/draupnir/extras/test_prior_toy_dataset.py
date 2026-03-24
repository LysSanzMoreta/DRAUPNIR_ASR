import torch
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
import numpy as np
import pyro
from pyro import distributions as dist

n_seqs = 500
z_dim = 30
X,y = make_blobs(n_samples=n_seqs, # number of observations
                n_features=z_dim, # number of features
                centers=5, # number of clusters
                cluster_std=0.5, # standard deviation of the clusters
                random_state=0)

train_idx = np.random.choice([True, False], size=n_seqs, p=[0.8, 0.2])
X_train , y_train = X[train_idx], y[train_idx]
X_test , y_test = X[~train_idx], y[~train_idx]

n_train = train_idx.sum()
init_params = dict(z_dim=z_dim,
                   n_leaves  = n_train,
                   n_leaves_batch = n_train,
                   n_leaves_internal_batch = n_seqs,
                   n_internal_batch = n_seqs - n_train,
                   )

full_patristic  = cdist(X, X, 'cosine')

train_patristic = full_patristic[train_idx][:,train_idx]
test_patristic = full_patristic[~train_idx][:,~train_idx]


"""
TODO: test covariance matrix
-----------------------------------------------------------------
Positive definiteness	All eigenvalues > 0 for every sampled covariance matrix
Prior predictive checks	Sample from prior alone (before seeing data) to ensure reasonable ranges
Posterior predictive checks	Compare simulated data to observed data (summary statistics, visualizations)
Coverage	Do 95% credible intervals contain ~95% of observed data?
Calibration	Are predictions well-calibrated on held-out test data?
-----------------------------------------------------------------
Check with the AI about different possibilities for the OU process

"""

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
    def __init__(self, sigma_f, lambd, sigma_n = None, kernel_type = "3"):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lambd = lambd
        self.kernel_type = kernel_type

    def prior1(self, t: torch.Tensor) -> torch.Tensor:

        assert self.sigma_f is not None, "sigma_f cannot be None"
        assert self.sigma_n is not None, "sigma_f cannot be None"

        lambd = self.lambd.unsqueeze(-1).unsqueeze(-1)  # self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1)  # [:,None,None]
        noise = torch.eye(t.shape[0])  # distributes noise/stochascity to diagonal of the covariance
        sigma_n = self.sigma_n.unsqueeze(-1).unsqueeze(-1)
        return first_term * second_term + sigma_n ** 2 * noise

    def prior2(self, t: torch.Tensor) -> torch.Tensor:

        assert self.sigma_f is not None, "sigma_f cannot be None"
        lambd = self.lambd.unsqueeze(-1).unsqueeze(-1)  # self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1)  # [:,None,None]
        noise = torch.eye(t.shape[0])  # distributes noise/stochascity to diagonal of the covariance

        return first_term * second_term + noise * 1e-6

    def prior3(self, t: torch.Tensor) -> torch.Tensor:

        lambd = self.lambd.unsqueeze(-1).unsqueeze(-1) #self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)
        return second_term

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "1":
            return self.prior1(t)
        elif self.kernel_type == "2":
            return self.prior2(t)
        elif self.kernel_type == "3":
            return self.prior3(t)


class TestModel():

    def __init__(self,init_params, test_patristic, tree_height):

        self.z_dim = init_params["z_dim"]
        self.n_leaves = init_params["n_leaves"]
        self.n_leaves_batch = init_params["n_leaves_batch"]
        self.n_leaves_internal_batch = init_params["n_leaves_internal_batch"]
        self.n_internal_batch = init_params["n_internal_batch"]
        self.internal_nodes_batch = test_patristic[0,1:]
        self.tree_height = tree_height



    def test_invertibility(self,matrix):
        #invertibility checkpoints:
        if torch.allclose(matrix,matrix.T, atol=1e-6):
            print("matrix is symmetric")
        else:
            print("matrix is not symmetric")
        eigvals = torch.linalg.eigvalsh(matrix)
        print("min eigenvalue per batch",eigvals.min(dim=-1).values) #have to be > 0


    def gp_prior_4(self,patristic_matrix_sorted):
        "Computes a Gaussian prior over the latent space. The Gaussian prior consists of a Ornstein - Ulenbeck kernel that uses the patristic distances to build a covariance matrix"

        #DUMMY SAMPLING!!!!!!!!!!!!!!!!!!!!

        rho = dist.Beta(8,2).sample() # correlation prior
        rho = rho.clamp(1e-8,1-1e-8)
        lambd = -2/torch.log(rho)

        patristic_matrix = patristic_matrix_sorted[1:, 1:] #[n_leaves_batch,n_leaves_batch]
        OU_covariance = OUKernel_Fast(None, lambd, None).forward(patristic_matrix) #[z_dim, n_leaves,n_leaves ]
        #assert OU_covariance.shape == (self.z_dim, self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        assert OU_covariance.shape == (self.n_leaves_batch, self.n_leaves_batch), f"Expected shape: ({self.z_dim},{self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        # #invertibility checkpoints:
        # if torch.allclose(OU_covariance,OU_covariance.T, atol=1e-6):
        #     print("matrix is symmetric")
        # else:
        #     print("matrix is not symmetric")
        # eigvals = torch.linalg.eigvalsh(OU_covariance)
        # print("min eigenvalue per batch",eigvals.min(dim=-1).values) #have to be > 0
        self.test_invertibility(OU_covariance)

        L = torch.linalg.cholesky(OU_covariance)
        eps_z = dist.Normal(0,1).sample([self.n_leaves_batch, self.z_dim]) #adds some noise to each of the leaves?
        latent_space = L @ eps_z
        assert latent_space.shape == (self.n_leaves_batch,self.z_dim)


        #latent_space = latent_space.T
        return {
                "eps_z": eps_z,
                "latent_space": latent_space,
                "rho": rho,
                "lambd": lambd
                }

    def gp_prior_5(self,patristic_matrix):

        D_max = patristic_matrix.max()
        log_lambd = pyro.sample("log_lambda", dist.Normal(torch.log(D_max / 2), 1.0).expand_by([1]))
        lambd = torch.exp(log_lambd)

        #lambd = DraupnirUtils.squeeze_tensor(1,lambd) #should not be necessary, fix

        OU_covariance = OUKernel_Fast(None, lambd, None).forward(patristic_matrix)  # [n_leaves,n_leaves ]
        #OU_covariance = DraupnirUtils.squeeze_tensor(2, OU_covariance)

        self.test_invertivility(OU_covariance)

        assert OU_covariance.shape == (self.n_leaves_batch,self.n_leaves_batch), f"Expected shape: ({self.n_leaves_batch},{self.n_leaves_batch}), got ({OU_covariance.shape})"
        L = torch.linalg.cholesky(OU_covariance)
        eps_z = pyro.sample("eps_z", dist.Normal(0, 1).expand_by([self.n_leaves_batch, self.z_dim]))  # adds some noise to each of the leaves?
        latent_space = L @ eps_z

        assert latent_space.shape == (self.n_leaves_batch,self.z_dim)

        return { "eps_z": eps_z,
                "latent_space": latent_space,
                "covariance": OU_covariance,
                "ou_params": {"lambd":lambd}
                }

    def conditional_sampling_4(self,map_estimates, patristic_matrix):
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

    def conditional_sampling_5(self,map_estimates, patristic_matrix):
            """Conditional sampling from Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)"""
            log_lambd = map_estimates["log_lambd"]  # + 1e-6
            lambd = torch.exp(log_lambd)

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











