import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import math
import gzip
import random
import tqdm
import numpy as np
from torch.nn import Linear, Identity, Module
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

class minGRU(Module):
    def __init__(self, dim, expansion_factor = 1., proj_out = None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)
        self.to_out = Linear(dim_inner, dim, bias = False) if proj_out else Identity()

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
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = self.log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
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

class minLM(Module):
    def __init__(
            self,
            *,
            num_tokens,
            dim,
            depth,
            ff_mult=4,
            expansion=1.5,
            conv_kernel_size=3,
            use_lstm=False,
            enable_conv=False,
            dropout=0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        min_rnn_klass = minGRU

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(dim, conv_kernel_size) if enable_conv else None,
                RMSNorm(dim),
                min_rnn_klass(dim, expansion_factor=expansion),
                RMSNorm(dim),
                FeedForward(dim, mult=ff_mult),
                nn.Dropout(dropout) if dropout > 0. else None
            ]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        self.can_cache = not enable_conv

    def forward(
            self,
            x,
            return_loss=False,
            return_prev_hiddens=False,
            prev_hiddens=None
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # handle previous hiddens, for recurrent decoding

        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        for conv, norm, mingru, ff_norm, ff, dropout in self.layers:

            # conv

            if exists(conv):
                assert len(list(prev_hiddens)) == 0, 'caching not supported for conv version'
                x = conv(x) + x

            # min gru

            prev_hidden = next(prev_hiddens, None)

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

            if exists(dropout):
                x = dropout(x)

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            if not return_prev_hiddens:
                return logits

            return logits, next_prev_hiddens

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss


NUM_BATCHES = int(1e2)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512


def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    prev_hiddens = None

    for _ in range(sample_num_times):
        logits, next_prev_hiddens = net(out, return_prev_hiddens = True, prev_hiddens = prev_hiddens)
        logits = logits[:, -1]

        if net.can_cache:
            prev_hiddens = next_prev_hiddens

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# accelerators

accelerator = Accelerator()

# the min{GRU|LSTM} char language model

model = minLM(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    use_lstm = False # set to True for minLSTM
)

# prepare enwik8 data

with gzip.open("data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)



class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(enumerate(train_loader))[1]

        loss = model(data, return_loss = True)

        accelerator.backward(loss / GRAD_ACCUM_EVERY)

    accelerator.print(f"training loss: {loss.item():.3f}")

    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(enumerate(val_loader))[1]

            loss = model(valid_data, return_loss = True)
            accelerator.print(f"validation loss: {loss.item():.3f}")

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.to(accelerator.device)

        prime = decode_tokens(inp)
        accelerator.print(f"INPUT: {prime}")

        prompt = inp[None, ...]

        sampled = base_decoding(model, prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        accelerator.print(f"\nOUTPUT: {base_decode_output}")
