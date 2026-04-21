
import torch
from abc import ABC, abstractmethod


class GPKernel(ABC):
    @abstractmethod
    def preforward(self, t1: torch.Tensor,t2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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

class OUKernel_Fast_old(GPKernel):
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
        """Not used function"""

        return torch.zeros((1,))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        #cov_b = torch.exp(-distance_matrix / _lambd) * _sigma_f ** 2 + _sigma_n + torch.eye(self.n_b*2, device=self.device) * 1e-5
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1) #[:,None,None]
        lamb = self.lamb.unsqueeze(-1).unsqueeze(-1) #self.lamb[:, None, None]
        second_term = torch.exp(-t / lamb)
        noise = torch.eye(t.shape[0]) #distributes noise/stochascity to diagonal of the covariance
        sigma_n = self.sigma_n.unsqueeze(-1).unsqueeze(-1)
        return first_term * second_term + sigma_n ** 2 * noise

class OUKernel_Fast(GPKernel):
    """ Kernel that computes the covariance matrix for a z Ornstein Ulenbeck processes. As stated in Equation 2.1 https://arxiv.org/pdf/1208.0628.pdf
    :param tensor sigma_f: Quantifies the intensity of inherited variation ---> Signal variance
    :param tensor lamb: Characteristic length-scale of the evolutionary dynamics (equivalent to the inverse of the strength of selection)---> Distance between data points (nodes),larger l implies that the noise should be bigger to capture big point fluctuations
    :param tensor sigma_n:quantifies the intensity of specific variation(i.e. variation unattributable to the phylogeny)--->Gaussian Noise,intensity of specific variation--> how much to let the sequence vary ---> so max branch lengh?
    **References:**
    "Ancestral Inference from Functional Data: Statistical Methods and Numerical Examples"
    """
    def __init__(self, sigma_f, lambd, sigma_n = None, z_dim = 30 , kernel_type = "3"):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.lambd = lambd
        self.kernel_type = kernel_type
        self.z_dim = z_dim

    def preforward(self,t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
            """Not used function"""

            return torch.zeros((1,))

    def kernel0(self, t: torch.Tensor) -> torch.Tensor:
        """OU kernel without hyperparameters"""

        assert self.sigma_f is None, "sigma_f should be None"
        assert self.lambd is None, "sigma_f should be None"
        assert self.sigma_n is None, "sigma_n should be None"

        t = t.repeat(self.z_dim,1,1)

        return torch.exp(-t)

    def kernel1(self, t: torch.Tensor) -> torch.Tensor: #original kernel
        "OU kernel with all 3 hyperparameters"
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

    def kernel2(self, t: torch.Tensor) -> torch.Tensor:
        "OU kernel with 2 hyperparameters, removed diagonal noise"

        assert self.sigma_f is not None, "sigma_f cannot be None"
        assert self.lambd is not None, "lambda cannot be None"

        lambd = self.lambd.unsqueeze(-1).unsqueeze(-1)  # self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)
        first_term = self.sigma_f ** 2
        first_term = first_term.unsqueeze(-1).unsqueeze(-1)  # [:,None,None]
        noise = torch.eye(t.shape[0])  # distributes noise/stochascity to diagonal of the covariance

        return first_term * second_term + noise * 1e-6

    def kernel3(self, t: torch.Tensor) -> torch.Tensor:
        "OU kernel with 1 hyperparameters"

        assert self.lambd is not None, "lambda cannot be None"
        assert self.sigma_f is None, "sigma_f has been removed, should be None"

        lambd = self.lambd.unsqueeze(-1)#.unsqueeze(-1) #self.lamb[:, None, None]
        second_term = torch.exp(-t / lambd)
        return second_term

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "0":
            return self.kernel0(t)
        elif self.kernel_type == "1":
            return self.kernel1(t)
        elif self.kernel_type == "2":
            return self.kernel2(t)
        elif self.kernel_type == "3":
            return self.kernel3(t)
        else:
            raise ValueError(f"{self.kernel_type} not found")

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