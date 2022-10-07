"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
# TORCH
import torch.nn as nn
import torch
from torch.distributions import constraints
import math
from ignite.engine import Engine, Events
from abc import ABC, abstractmethod
from typing import Callable
#Pyro
from pyro.infer import SVI
from pyro import distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution



class RNNEncoder(nn.Module):
    def __init__(self, align_seq_len,aa_prob,n_leaves,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNEncoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.rnn_input_size = rnn_input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.linear_means = nn.Linear(self.gru_hidden_dim, self.z_dim)
        self.linear_std = nn.Linear(self.gru_hidden_dim, self.z_dim)

        self.softplus = nn.Softplus()

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)


    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        rnn_output = self.fc1(rnn_output[:,-1]) #pick the last state of the sequence given by the GRU
        output_means = self.linear_means(rnn_output)
        output_std = self.softplus(self.linear_std(rnn_output))
        return output_means,output_std
class RNNEncoder_no_mean(nn.Module):
    def __init__(self, align_seq_len,aa_prob,n_leaves,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNEncoder_no_mean, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.rnn_input_size = rnn_input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        #self.linear_means = nn.Linear(self.gru_hidden_dim, self.z_dim)
        self.linear_std = nn.Linear(self.gru_hidden_dim, self.z_dim)

        self.softplus = nn.Softplus()

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)


    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        rnn_output = self.fc1(rnn_output[:,-1]) #pick the last state of the sequence given by the GRU
        #output_means = self.linear_means(rnn_output)
        output_std = self.softplus(self.linear_std(rnn_output))
        return output_std
class RNNDecoder_Tiling(nn.Module):
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)
            #rnn.state_dict()


    def forward(self, input, hidden):
        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output_logits = self.logsoftmax(self.linear_probs(self.fc1(rnn_output)))  # [n_nodes,align_seq_len,aa_probs]
        return output_logits
class RNNDecoder_Tiling_Angles(nn.Module):
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling_Angles, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = 4
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.fc2_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
        self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
        self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output = self.fc1(rnn_output)
        output_logits = self.logsoftmax(self.fc2_probs((output)))  # [n_nodes,align_seq_len,aa_probs]
        output_means = self.tanh(self.fc2_means(output))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(output))
        return output_logits,output_means,output_kappas
class RNNDecoder_Tiling_AnglesComplex(nn.Module):
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,rnn_input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling_AnglesComplex, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.rnn_input_size = rnn_input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.fc1_probs = nn.Linear(self.gru_hidden_dim,int((self.gru_hidden_dim)//2))
        self.fc1_means = nn.Linear(self.gru_hidden_dim, int((self.gru_hidden_dim) // 2))
        self.fc1_kappas = nn.Linear(self.gru_hidden_dim, int((self.gru_hidden_dim) // 2))
        self.fc2_probs = nn.Linear(int((self.gru_hidden_dim) // 2), self.aa_prob)
        self.fc2_means = nn.Linear(int((self.gru_hidden_dim) // 2), 2)
        self.fc2_kappas = nn.Linear(int((self.gru_hidden_dim) // 2), 2)
        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output = self.fc1(rnn_output)
        output_logits = self.logsoftmax(self.fc2_probs(self.fc1_probs(output)))  # [n_nodes,align_seq_len,aa_probs]
        output_means = self.tanh(self.fc2_means(self.fc1_means(output)))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(self.fc1_kappas(output)))
        return output_logits,output_means,output_kappas

class Embed(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(Embed, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.aa_probs,self.aa_probs)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        return output
class EmbedComplex(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(EmbedComplex, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.aa_probs,self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim,self.aa_probs)
    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.softmax(self.fc2(output))
        return output

class EmbedComplexEncoder(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(EmbedComplexEncoder, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.aa_probs,self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim,self.aa_probs)

    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.softmax(self.fc2(output))
        return output


class GPKernel(ABC):
    @abstractmethod
    def preforward(self, t1: torch.Tensor,t2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
class SVIEngine(Engine):
    def __init__(self, *args, step_args=None, **kwargs):
        self.svi = SVI(*args, **kwargs)
        self._step_args = step_args or {}
        super(SVIEngine, self).__init__(self._update)

    def _update(self, engine, batch):
        return -engine.svi.step(batch, **self._step_args)
class OUKernel_SimulationFunctionalValuesTraits(GPKernel):
    """ Kernel that computes the covariance matrix for a z Ornstein Ulenbeck processes. As stated in Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
    :param tensor sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
    :param tensor lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes),larger l implies that the noise should be bigger to capture big point fluctuations
    :param tensor sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
    **References:**
    "Ancestral Inference from Functional Data: Statistical Methods and Numerical Examples"
    """
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb

    def preforward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        first_term = self.sigma_f ** 2
        second_term = torch.exp(-t / self.lamb)
        return first_term * second_term + self.sigma_n ** 2 * torch.eye(t.shape[0])

class OUKernel_Fast(GPKernel):
    """ Kernel that computes the covariance matrix for a z Ornstein Ulenbeck processes. As stated in Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
    :param tensor sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
    :param tensor lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes),larger l implies that the noise should be bigger to capture big point fluctuations
    :param tensor sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
    **References:**
    "Ancestral Inference from Functional Data: Statistical Methods and Numerical Examples"
    """
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb
    def preforward(self,t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = t1.unsqueeze(1) - t2.unsqueeze(0)
        absdiff = diff.abs().sum(-1)
        return absdiff

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        #cov_b = torch.exp(-distance_matrix / _lambd) * _sigma_f ** 2 + _sigma_n + torch.eye(self.n_b*2, device=self.device) * 1e-5
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1)
        lamb = self.lamb.unsqueeze(-1).unsqueeze(-1) #self.lamb[:, None, None]
        second_term = torch.exp(-t / lamb)
        noise = torch.eye(t.shape[0]) #distributes noise/stochascity to diagonal of the covariance
        sigma_n = self.sigma_n.unsqueeze(-1).unsqueeze(-1)
        return first_term * second_term + sigma_n ** 2 * noise
class OUKernel_Fast_Sparse(GPKernel):
    """ Kernel that computes the covariance matrix for a z Ornstein Ulenbeck processes, in this case for a sparse Gaussian process. As stated in Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
    :param tensor sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
    :param tensor lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes),larger l implies that the noise should be bigger to capture big point fluctuations
    :param tensor sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
    """
    def __init__(self, sigma_f, sigma_n, lamb):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lamb = lamb
    def preforward(self,t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = t1.unsqueeze(1) - t2.unsqueeze(0)
        absdiff = diff.abs().sum(-1)
        return absdiff
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        first_term = self.sigma_f ** 2
        second_term = torch.exp(-t / self.lamb[:, None, None])
        return first_term[:, None, None] * second_term + self.sigma_n[:, None, None] ** 2
class VSGP(TorchDistribution):
    """
    Variational Sparse Gaussian Process distribution
    Follow this if anything is missing: https://github.com/pytorch/pytorch/blob/21c2542b6a9faafce0b6a3e1583a07b3fba9269d/torch/distributions/multivariate_normal.py
    """

    def __init__(self, kernel: GPKernel, inducing_set: torch.Tensor,
                 output_distribution_f: Callable[[torch.Tensor, torch.Tensor], dist.Distribution],
                 *, input_data: torch.Tensor, eps=1e-1):
        super().__init__()
        arg_constraints = {'loc': constraints.real_vector,
                           'covariance_matrix': constraints.positive_definite,
                           'precision_matrix': constraints.positive_definite,
                           'scale_tril': constraints.lower_cholesky}
        self.kernel = kernel
        self.inducing_set = inducing_set #selects a bunch of nodes from a Normal distribution
        self.output_distribution_f = output_distribution_f
        self.input_data = input_data #patristic distances
        self._eps = eps
        self._compute_out_dist()
        self.support = constraints.real_vector
        #batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]

    def _compute_out_dist(self):
        induced_induced = self.kernel.preforward(self.inducing_set.unsqueeze(-1),self.inducing_set.unsqueeze(-1))
        kmm = self.kernel.forward(induced_induced) #[z_dim,n_inducing_points,n_inducing_points]
        noise = torch.eye(kmm.size()[1]) * self._eps #[n_inducing_points,n_inducing_points]
        kmm = kmm + noise[None,:,:] #TODO: We need to add the noise outside, because of shape problems
        input_induced = self.kernel.preforward(self.input_data,self.inducing_set.unsqueeze(-1)) #[n_seq,n_inducing points]
        knm = self.kernel.forward(input_induced) #[n_seq,n_nodes,n_inducing_points]
        kmm_inv = kmm.inverse() #[z_dim,n_inducing_points,n_inducing_points]
        kmn = knm.transpose(-1, -2) #[z_dim,n_inducing_points,n_seq]
        input_input = self.kernel.preforward(self.input_data,self.input_data) #[n_seq,n_seq]
        knn = self.kernel.forward(input_input) ##[z_dim,n_seq,n_seq]
        #self._f_mean = knm @ kmm_inv @ self.inducing_set #[z_dim,n_seq] #TODO: What is this?
        self._f_mean = torch.zeros([self.input_data.shape[0]])

        self._f_var = knn - knm @ kmm_inv @ kmn ##[z_dim,n_seq,n_seq]
        self._out_dist = self.output_distribution_f(self._f_mean[None, :], self._f_var)

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def conjugate_update(self, other):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        return self._out_dist.sample(*args, **kwargs)

    def log_prob(self, x, *args, **kwargs):
        return self._out_dist.log_prob(x, *args, **kwargs)

    def support(self):
        """
        Returns a :class:`~torch.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
        return self.support()

def masking(dataset):
    """Creating a mask for the gaps (0) in the data set"""
    mask_indx = dataset.eq(0)  # Find where is equal to 0 == gap
    dataset_mask = torch.ones(dataset.shape)
    dataset_mask[mask_indx] = 0
    return dataset_mask
def print_divisors(n) :
    """Calculates the number of divisors of a number
    :param int n: number"""
    i = 1
    divisors = []
    while i <= n :
        if (n % i==0) :
            divisors.append(i)
        i = i + 1
    return divisors
def intervals(parts, duration):
    """Compose a list of intervals on which a number is divided """
    part_duration = duration / parts
    return [(int(i) * part_duration, (int(i) + 1) * part_duration) for i in range(parts)]
def compute_sites_entropies(logits, node_names):
    """
    Calculate the Shannon entropy of a sequence
    :param tensor logits = [n_seq, L, 21]
    :param tensor node_names: tensor with the nodes tree level order indexes ("names")
    observed = [n_seq,L]
    Pick the aa with the highest logit,
    logits = log(prob/1-prob)
    prob = exp(logit)/(1+exp(logit))
    entropy = prob.log(prob) per position in the sequence
    The entropy H is maximal when each of the symbols in the position has equal probability
    The entropy H is minimal when one of the symbols has probability 1 and the rest 0. H = 0"""
    #probs = torch.exp(logits)  # torch.sum(probs,dim=2) returns 1's so , it's correct

    prob = torch.exp(logits) / (1 + torch.exp(logits))
    seq_entropies = -torch.sum(prob*torch.log(prob),dim=2)

    seq_entropies = torch.cat((node_names[:,None],seq_entropies),dim=1)
    return seq_entropies
def compute_seq_probabilities(logits, observed,train=True):
    """Compute the sequence probabilities (prob = exp(logit)/(1+exp(logit))) from the logits
    :param tensor logits: log(prob/1-prob), [n_seq, L, 21]
    :param tensor observed = [n_seq,L]
    """
    #probs = torch.exp(logits)  # torch.sum(probs,dim=2) returns 1's so , it's correct
    node_names = observed[:, 0, 1]
    aminoacids = observed[:,2:,0]
    prob = torch.exp(logits) / (1 + torch.exp(logits))
    if train:
        prob_max = prob.gather(2, aminoacids[:, :, None]) #pick the probability corresponding to the observed aminoacid

    else: #for the test we "do not have" the observed sequences, so we use the highest logits as a reference
        prob_argmax = torch.argmax(prob,dim=2)
        prob_max = prob.gather(2,prob_argmax[:,:,None])
    print("min prob {}".format(torch.min(prob_max)))
    print("max prob {}".format(torch.max(prob_max)))
    #seq_probabilities = torch.sum(torch.log(prob_max), 1) #transform into log not to lose information in the product
    seq_probabilities = torch.prod(prob_max, 1)  # transform into log not to lose information in the product

    seq_probabilities = torch.cat((node_names[:,None],seq_probabilities),dim=1)
    return seq_probabilities


