"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
# TORCH
import torch.nn as nn
import torch
import  torch.nn.functional as F
from torch.distributions import constraints
import math
from ignite.engine import Engine, Events
from abc import ABC, abstractmethod
from typing import Callable
#Pyro
from pyro.infer import SVI
from pyro import distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from torch.nn import Module, ModuleList, RMSNorm



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

class EmbedComplex_1b(nn.Module):
    def __init__(self,aa_probs,embedding_dim,pretrained_params):
        super(EmbedComplex_1b, self).__init__()
        self.aa_probs = aa_probs
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.aa_probs,self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim,self.aa_probs)
        self.layernorm1 = nn.LayerNorm(self.embedding_dim)
        self.layernorm2 = nn.LayerNorm(self.aa_probs)

    def forward(self,input):

        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.layernorm1(output)
        output = self.softmax(self.layernorm2(self.fc2(output)))

        return output

class EmbedComplexEncoder(nn.Module):
    def __init__(self,input_dim,embedding_dim,out_dim):
        super(EmbedComplexEncoder, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.input_dim,self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim,self.out_dim)

    def forward(self,input):

        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.softmax(self.fc2(output))

        return output

class EmbedComplexEncoder_1b(nn.Module):
    def __init__(self,input_dim,embedding_dim,out_dim):
        super(EmbedComplexEncoder_1b, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.input_dim,self.embedding_dim)
        self.layernorm1 = nn.LayerNorm(self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim,self.out_dim)
        self.layernorm2 = nn.LayerNorm(self.out_dim)

    def forward(self,input):
        output = self.fc1(input) #.type(torch.cuda.IntTensor)
        output = self.layernorm1(output)
        output = self.softmax(self.layernorm2(self.fc2(output)))

        return output

class FCFilm(nn.Module):
    def __init__(self, input_dim, embedding_dim, out_dim):
        super(FCFilm, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.Softmax(dim=-1)
        )
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, input, latent_space):
        """
        Transform the hidden states conditioned by the latent space
        :input: [N, L , feat_dim] , hidden states
        :latent_space : [N, z_dim]
        """

        gamma_beta = self.mlp(latent_space)
        gamma, beta = gamma_beta.chunk(2, dim=-1) #split into 2 chunks

        gamma = gamma.unsqueeze(1)           # (N, 1, featdim)
        beta  = beta.unsqueeze(1)            # (N, 1, featdim)

        output = gamma * input + beta

        output = self.logsoftmax(output)

        return output

class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value=0.)
        x = self.net(x)
        return x.transpose(1, 2)  # b d n -> b n d

class minGRU(Module):
    def __init__(self, dim, expansion_factor = 1., proj_out = None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = proj_out if proj_out is not None else expansion_factor != 1.

        self.to_hidden_and_gate = torch.nn.Linear(dim, dim_inner * 2, bias = False)
        self.to_out = torch.nn.Linear(dim_inner, dim, bias = False) if proj_out else torch.nn.Identity()

    def g(self,x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def log_g(self,x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def heinsen_associative_scan_log(self,log_coeffs, log_values):
        """https://github.com/glassroom/heinsen_sequence"""
        a_star = log_coeffs.cumsum(dim=1)
        log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
        log_h = a_star + log_h0_plus_b_star
        return log_h.exp()

    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            # handle sequential

            hidden = self.g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if prev_hidden is not None else (hidden * gate) #input=prev_hidde, end=hiddeh, weight = gate
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = self.log_g(hidden)
            log_values = log_z + log_tilde_h

            if prev_hidden is not None:
                log_values = torch.cat((prev_hidden.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = self.heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]
        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden

def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

class miniGRUEncoder(nn.Module):
    def __init__(self,
                 depth,
                 input_dim,
                 output_dim,
                 enable_conv=False,
                 conv_kernel_size=3,
                 expansion=1.5,
                 ff_mult=4,
                 dropout=0
                 ):
      super(miniGRUEncoder, self).__init__()
      self.input_dim = input_dim
      self.output_dim = output_dim
      self.layers = ModuleList([])
      for _ in range(depth):
          self.layers.append(ModuleList([
              CausalDepthWiseConv1d(self.input_dim, conv_kernel_size) if enable_conv else None,
              RMSNorm(self.input_dim),
              minGRU(self.input_dim, expansion_factor=expansion),
              RMSNorm(self.input_dim),
              FeedForward(self.input_dim, mult=ff_mult),
              nn.Dropout(dropout) if dropout > 0. else None
          ]))

      self.norm = RMSNorm(self.input_dim)
      self.logits_probs = nn.Linear(self.input_dim, self.output_dim)
      self.linear_means = nn.Linear(self.input_dim, self.output_dim)
      self.linear_std = nn.Linear(self.input_dim, self.output_dim)
      self.softplus = nn.Softplus()

    def forward(self,
                x,
                prev_hiddens=None):
        if prev_hiddens is not None:
            x = x[:, -1:]
        next_prev_hiddens = []
        prev_hiddens = prev_hiddens if prev_hiddens is not None else []

        for conv, norm, mingru, ff_norm, ff, dropout in self.layers:
            # conv
            if conv is not None:
                assert len(list(prev_hiddens)) == 0, 'caching not supported for conv version'
                x = conv(x) + x
            # min gru
            prev_hidden = next(iter(prev_hiddens), None)
            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden=True
            )

            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)

            # feedforward
            x = ff(ff_norm(x)) + x
            # dropout
            if dropout is not None:
                x = dropout(x)

        embed = self.norm(x)
        logits = self.logits_probs(embed)

        latent = x.mean(dim=1)

        z_loc = self.linear_means(latent)
        z_scale = self.softplus(self.linear_std(latent))

        return {"prev_hiddens": prev_hiddens, #todo: do something with this? not used in the example
                "embeddings":logits,
                "z_loc":z_loc,
                "z_scale":z_scale
                }

class miniGRUDecoder(nn.Module):
    def __init__(self,
                 depth,
                 input_dim,
                 output_dim,
                 enable_conv=False,
                 conv_kernel_size=3,
                 expansion=1.5,
                 ff_mult=4,
                 dropout=0
                 ):
      super(miniGRUDecoder, self).__init__()
      self.input_dim = input_dim
      self.output_dim = output_dim
      self.layers = ModuleList([])
      for _ in range(depth):
          self.layers.append(ModuleList([
              CausalDepthWiseConv1d(self.input_dim, conv_kernel_size) if enable_conv else None,
              RMSNorm(self.input_dim),
              minGRU(self.input_dim, expansion_factor=expansion),
              RMSNorm(self.input_dim),
              FeedForward(self.input_dim, mult=ff_mult),
              nn.Dropout(dropout) if dropout > 0. else None
          ]))
      self.norm = RMSNorm(self.input_dim)
      self.logits_probs = nn.Linear(self.input_dim,self.output_dim)
      self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self,
                x,
                prev_hiddens=None):
        if prev_hiddens is not None:
            x = x[:, -1:]
        next_prev_hiddens = []
        prev_hiddens = prev_hiddens if prev_hiddens is not None else []

        for conv, norm, mingru, ff_norm, ff, dropout in self.layers:
            # conv
            if conv is not None:
                assert len(list(prev_hiddens)) == 0, 'caching not supported for conv version'
                x = conv(x) + x
            # min gru
            prev_hidden = next(iter(prev_hiddens), None)
            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden=True
            )

            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)

            # feedforward
            x = ff(ff_norm(x)) + x
            # dropout
            if dropout is not None:
                x = dropout(x)

        embed = self.norm(x)
        logits = self.logsoftmax(self.logits_probs(embed))

        return logits

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

class PositionalEncodings(nn.Module):
    def __init__(self,max_seq_len,feat_dim,base=1000, type="sinusoidal"):
        super(PositionalEncodings, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.max_seq_len = max_seq_len
        self.feat_dim = feat_dim
        self.base = base
        self.positions_idx = torch.arange(0, self.max_seq_len)[None, :]  # [1,L]
        self.dimensions_idx = torch.arange(0, self.feat_dim // 2 )
        self.type = type

    def align(self,tensor, axes, ndim=None): #TODO: Do I need this? Does not seem to do anything
        """Expand dimensions
        axes：
        ndim：
        """
        assert len(axes) == tensor.ndim
        assert ndim or min(axes) >= 0
        ndim = ndim or max(axes) + 1
        indices = [None] * ndim
        for i in axes:
            indices[i] = slice(None)
        return tensor[indices]
    def sinusoidal_encodings(self):
        """
        Implements trigonometric or sinusoidal positional encodings. Identical results to those in the annotated transformer

        1) Calculation of frequencies: feat_dim=10, base=10000
        MethodA : torch.pow(self.base , -2 * torch.arange(0,self.feat_dim//2) / self.feat_dim)
        [0,1,2,3,4] -> divide by dim ->  [0,0,1,0.2,0.3,0.4] -> [0,-0.2,-0.4,-0.6,-0.8] -> power -> [10000⁰, 10000^{-0.2}, 10000^{-0.4},10000^{-0.6}, 10000^{-0.8}] (already inverted)
        MethodB:  1/ (self.base**(torch.arange(0,self.feat_dim,2)/self.feat_dim))
        [0,2,4,6,8] -> divide by dim -> [0,0.2,0.4,0.6,0.8] --> power -> [10000⁰, 10000^{0.2}, 10000^{0.4},10000^{0.6}, 10000^{0.8}] -> invert -> [1/10000⁰, 1/10000^{0.2}, ....]

        2) Vector multiplication via Einstein summation:

        y = a1x1 + a2x2 + ... + anxn ->    y = sum(aixi) -> ii

        #Example1: Selects the diagonal values (00,11,22) (the trace) and sums them so the result is 15 (1 + 5 + 9)
        # >>> torch.einsum("ii",torch.tensor([[1,2,3],[4,5,6],[7,8,9]]))
        # 15
        #Example2: 1D vector multiplication: c00 = a0*b0; c01 = a0*b1 ....
        # >>> a = torch.tensor([3,4,5])
        # >>> b = torch.tensor([6,7,8])
        # >>> torch.einsum("i,j->ij",a,b)
        tensor([[18, 21, 24],
                [24, 28, 32],
                [30, 35, 40]])
        # The above is equivalent to torch.einsum("...,i->...i",a,b) and to torch.matmul(a[:,None],b[None,:])

        3) Stacking the sine and cosine results in an overlapping manner:

        Given the frequencies for a single feature vector of size 3 (feat_dim) at time step 0 of the sequence
        frequencies  = [\theta1, \theta2, \theta3]
        sine_frequencies = [sin\theta1, sin\theta2, sin\theta3]
        cosine_frequencies = [cos\theta1, cos\theta2, cos\theta3]

        stack = torch.stack([sine_frequencies,cosine_frequencies],dim=-2)
        stack
             [[sin\theta1,cos\theta1],
              [sin\theta2,cos\theta2],
              [sin\theta3,cos\theta3]]
        stack.flatten(-2)
             [sin\theta1,cos\theta1,sin\theta2,cos\theta2,sin\theta3,cos\theta3]
        when we index the odd positions of the flattened stack latter, we will get the values for the cosine and fof the even, the sine

        SOURCE: https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L845

        returns:
            :param sinusoidal_embeddings of shape [L,feat_dim*2]

        """

        frequencies= torch.pow(self.base , -2 * self.dimensions_idx / self.feat_dim) #frequencies or angle rates #[feat_dim]
        #frequencies = 1/ (self.base**(torch.arange(0,self.feat_dim,2)/self.feat_dim)) #equivalent code, but needs one more operation
        frequencies = torch.einsum('...,d->...d', self.positions_idx,frequencies) #1D vector multiplication [L,feat_dim] #equivalent to torch.einsum('i,j->ij') or torch.matmul(a[:,None],b[None,:])
        #Highlight: The upcoming code is a bit of a cumbersome one and I am not sure why they do not just keep the cosine and sine values separated always,
        # and avoid indexing. I do not think they are ever used together...
        sinusoidal_embeddings = torch.stack([torch.sin(frequencies), torch.cos(frequencies)], axis=-1) #[1,L,feat_dim//2,2]
        sinusoidal_embeddings = torch.flatten(sinusoidal_embeddings, -2) #[1,L,feat_dim]

        return sinusoidal_embeddings
    def apply_rotation(self,sinusoidal_embedding, inputs):
        """
        Implements the necessary steps to prepare the tensors for rotation as in Equation 34 in the paper https://arxiv.org/pdf/2104.09864
        1) Split the hidden dimensions from the inputs (key and query) into chunks of 2

        2) Rotation in 2D (x \in R^2 for the feature dimension)

        W = [[W00,W01],       X = [X0,X1]  ----->  X' = W@X = [[W00.X0 + W01.X1], = [X0' , X1']
             [W10,W11]]                                    [W10.X0 + W10.X1]]
        R = [[cosm\theta, -sinm\theta], ------>  R@X' = [[cosm\theta1.X0', -sinm\theta1.X0'], ---------->Split up ----------> R@X' = [[cosm\theta1] *[[x0], + [[sinm\theta1] *[[-x1],
             [sinm\theta, cosm\theta]]                 [sinm\theta1.X1', cosm\theta1.X1']]                                          [cosm\theta1]] [x1]]     [sinm\theta1]   [x0]]

        3) Rotation in n-dimensions

        R@X' = [[cosm\theta1], *[[x0], + [[sinm\theta1], *[[-x1],
                [cosm\theta1],  [x1],     [sinm\theta1],   [x0],
                [cosm\theta2],  [x2],     [sinm\theta2],   [x3],
                [cosm\theta2]]  [x3]]     [sinm\theta2]]   [x2]]

        Implements adjacent rope rank, similar to https://github.com/Tongjilibo/bert4torch/blob/ce644b9cefa72801a4a23366cb2cd9b895511951/bert4torch/layers/position_encoding.py#L170-L256C6
        return
            :param inputs of shape [N,L,...,feat_dim], where L is the max length. Inputs can be the keys and queries
        """
        ndim = inputs[0].ndim
        sinusoidal_embedding = self.align(sinusoidal_embedding,[0, 1, -1], ndim) #[1,L,featdim]
        cos_positions = torch.repeat_interleave(sinusoidal_embedding[...,1::2],2,-1) #We need to duplicate the frequencies so that we can perform the rotation in nD (\theta1,\theta1,\theta2,\theta2) #repeats the last dimension of the tensor such that [0,1,2,3,4,5] -> [0,0,1,1,2,2,3,3]
        sine_positions = torch.repeat_interleave(sinusoidal_embedding[...,::2],2,-1)

        #Highlight: from https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py#L359
        rotated_outputs = []
        for x in inputs:
            tensor2 = torch.stack([-x[..., 1::2], x[..., ::2]], ndim) #Highlight: need to use ndim, no 2
            tensor2 = torch.reshape(tensor2, x.shape) #these 2 lines turn -tensor2- into [-x2,x1,-x4,x3, ....]
            # x1,x2 = x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:] #This comes from the bert2torch implementation and it is not correct in this setting perhaps it needs a different context
            # tensor2 = torch.cat([-x2,x1], dim=x1.ndim - 1)
            rotated_outputs.append(x * cos_positions + tensor2 * sine_positions) #Equation 34

        return rotated_outputs
    def forward(self,input):

        sinusoidal_embeddings = self.sinusoidal_encodings()  # independent of the input content except for the maxlen and featdim
        if self.type == "sinusoidal":

            self.register_buffer("positional_encoding", sinusoidal_embeddings)
            input = input + sinusoidal_embeddings[:, : input.size(1)].requires_grad_(False)

            return input

        elif self.type == "rotary":
            # key = self.query_fc(input)
            # query = self.key_fc(input)
            key = input
            query = input
            #values = input

            kw, qw = self.apply_rotation(sinusoidal_embeddings, [key, query]) #TODO: WARNING : Switched qw,kw

            return kw,qw

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
    part_duration = int(duration / parts)

    return [(int(i * part_duration), int((int(i) + 1) * part_duration)) for i in range(parts)]
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

    prob_softmax = torch.nn.Softmax(dim=-1)(logits)

    #prob_sigmoid = torch.exp(logits) / (1 + torch.exp(logits)) #kind of sigmoid function
    seq_entropies = -torch.sum(prob_softmax*torch.log(prob_softmax),dim=2)

    seq_entropies = torch.cat((node_names[:,None],seq_entropies),dim=1)
    node_names_2d = node_names[:,None].tile((logits.shape[2])).reshape(len(node_names),1,logits.shape[2])

    prob_softmax = torch.concatenate((node_names_2d,prob_softmax),dim=1)

    return seq_entropies, prob_softmax
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


