import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class Encoder(torch.nn.Module):
    def __init__(self, L, D, z_dim):
        super(Encoder,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(L * D, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * z_dim)
        )
        self.z_dim = z_dim

    def forward(self, x):
        # x: [N, L, D]
        N = x.shape[0]
        x_flat = x.reshape(N, -1)


        stats = self.net(x_flat)
        loc, log_scale = stats.chunk(2, dim=-1)
        scale = torch.exp(log_scale)
        return loc, scale

class Decoder(torch.nn.Module):
    def __init__(self, L, z_dim, K):
        super(Decoder,self).__init__()
        self.L = L
        self.K = K

        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, L * K)
        )

        # Potts coupling (shared across positions)

        #self.J = PyroParam(torch.randn(K, K) * 0.01)
        self.J = nn.Parameter(torch.randn(K, K) * 0.01)

    def forward(self, z):
        # z: [N, z_dim]
        N = z.shape[0]
        logits = self.net(z).reshape(N, self.L, self.K)
        return logits, self.J

class PottsVAE(nn.Module):
    def __init__(self, encoder, decoder, L, K, z_dim):
        super(PottsVAE,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.L = L
        self.K = K #feat_dim?
        self.z_dim = z_dim

    def potts_pseudolikelihood(self,x, logits, J):
        """
        x: [N, L] (integer states 0..K-1)
        logits: [N, L, K]
        J: [K, K]
        """
        N, L = x.shape[0], x.shape[1]
        K = logits.shape[-1]

        log_prob = 0.0

        for l in range(L):
            # local logits
            local = logits[:, l, :]  # [N, feat_dim]
            # pairwise contribution
            pairwise = 0.0 #observed pairwise frequency
            for l2 in range(L):
                if l2 == l:
                    continue
                x_l2 = x[:, l2].squeeze(-1).int()  # [N]

                print(pairwise)

                print(J.shape)

                print(x_l2)



                pairwise = pairwise + J[:, x_l2].T  # [N, K]

            total_logits = local + pairwise

            log_prob = log_prob + dist.Categorical(logits=total_logits).log_prob(x[:, l])

        return log_prob.sum()

    def model(self, x):
        #pyro.module("decoder", self.decoder)

        N , L = x.shape[0], x.shape[1]

        with pyro.plate("data", N):
            z = pyro.sample("z", dist.Normal(0, 1).expand([self.z_dim]).to_event(1))

            logits, J = self.decoder(L, z_dim, K)(z)

            logp = self.potts_pseudolikelihood(x, logits, J)

            pyro.factor("potts_likelihood", logp)

    def guide(self, x):
        #pyro.module("encoder", self.encoder)

        N, L, K  = x.shape

        with pyro.plate("data", N):
            loc, scale = self.encoder(L, K,z_dim)(x)

            pyro.sample("z", dist.Normal(loc, scale).to_event(1))


train_data = torch.randint(low=0,high=20,size=(50,10,1)).float() #todo: substitute for fibonacci
N,L,_ = train_data.shape
z_dim = 5
K = 3 # hidden dim
vae = PottsVAE(Encoder, Decoder, L, K, z_dim)

optimizer = Adam({"lr": 1e-3})
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

for step in range(5):
    loss = svi.step(train_data)
    print("loss", loss)
    if step % 100 == 0:
        print(step, loss)


"""

https://www.youtube.com/watch?v=2Aw6RkzwGmg
https://github.com/HussainAther/potts

https://web.stanford.edu/group/candes/metro/potts_example_py.html

https://www.nature.com/articles/s41467-021-26529-9

https://github.com/xqding/DCA

https://github.com/kminartz/NeuralCPM

"""