"""
VAE for protein sequences with frozen ESM-2 encoder and ProtGPT2 decoder
=======================================================================

Model overview
--------------

We build a variational autoencoder on top of two large pretrained models:

  • Encoder:
      x (amino-acid sequence) --tokenize--> ESM-2
      ESM-2 last_hidden_state (B, T_enc, H_esm)
      ↓ mean-pooling over valid positions
      h (B, H_esm)
      ↓ small MLP
      q(z | x) = Normal(mu(x), diag(sigma^2(x))) in R^{z_dim}

  • Prior:
      p(z) = Normal(0, I) in R^{z_dim}
      KL scaled by a warmup factor over epochs.

  • Decoder:
      z --small adapter MLP--> soft prefix embeddings (B, L_prefix, H_gpt)
      prefix used as a continuous "prompt" for ProtGPT2 (frozen LM)

      For training:
        - We use teacher forcing on the target sequence tokens.
        - At each time step t, the LM predicts token x_t given prefix and x_{<t}.
        - The likelihood is a masked categorical distribution over token indices.

      For generation:
        - Sample z ~ N(0, I).
        - Map z to soft prefix embeddings.
        - Run ProtGPT2.generate with inputs_embeds=prefix and an explicit attention_mask.
        - Decode the generated token sequences to amino-acid strings.

Plates and random variables
---------------------------

  • Plate "batch_z" over sequences (size B) for the latent variable z:
        with pyro.plate("batch_z", B, dim=-1):
            z ~ Normal(0, I)             # model
            z ~ Normal(mu(x), sigma(x))  # guide

    The event shape of z is (z_dim,). The batch shape is (B,).

  • Plates "batch" and "time" for the token likelihood:
        logits: Categorical over tokens with batch_shape (B, L) and event_shape ()
        mask:   (B, L) boolean, True where a token contributes to the likelihood.

        with pyro.plate("batch", B, dim=-2):
            with pyro.plate("time", L, dim=-1):
                x_obs ~ MaskedDistribution(Categorical(logits), mask)

Labels are always valid token IDs; the mask determines which positions contribute.

Conda environment setup
-----------------------

These commands assume a recent CUDA-capable system; adjust CUDA version if needed.
For CPU-only, you can install PyTorch without CUDA.

1. Create and activate a new environment:

    conda create -n protvae python=3.10 -y
    conda activate protvae

2. Install PyTorch (CUDA example for Linux, CUDA 12.1):

    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

   For CPU-only, you can do:

    conda install pytorch cpuonly -c pytorch -y

3. Install the remaining Python packages:

    pip install -U transformers safetensors pyro-ppl sentencepiece einops tqdm

4. Run this script:

    python vae_esm_protgpt.py

"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import pyro.poutine as poutine

from transformers import AutoTokenizer, EsmModel, GPT2TokenizerFast, GPT2LMHeadModel

print("Done imports")
# ---------------------------------------------------------------------------
# Utility functions and configuration
# ---------------------------------------------------------------------------

def assert_shape(t: torch.Tensor, *expected: int, name: str = "tensor") -> None:
    """Check that a tensor has exactly the expected shape."""
    s, e = tuple(t.shape), tuple(expected)
    assert s == e, f"{name} shape {s} != expected {e}"

def assert_lastdim(t: torch.Tensor, expected: int, name: str = "tensor") -> None:
    """Check that a tensor's last dimension has the expected size."""
    s = tuple(t.shape)
    assert s and s[-1] == expected, f"{name} lastdim {s[-1]} != {expected}; shape={s}"

def assert_same_device(*tensors: torch.Tensor) -> None:
    """Ensure all tensors live on the same device."""
    devs = {str(t.device) for t in tensors if isinstance(t, torch.Tensor)}
    assert len(devs) == 1, f"Device mismatch: {devs}"

def assert_bool_mask(mask: torch.Tensor, ref: torch.Tensor, name: str = "mask") -> None:
    """Validate that mask is boolean and shape-compatible with ref."""
    assert mask.dtype == torch.bool, f"{name} must be bool, got {mask.dtype}"
    assert mask.shape == ref.shape, f"{name} shape {tuple(mask.shape)} != ref {tuple(ref.shape)}"

def assert_int_labels(labels: torch.Tensor, ref: torch.Tensor, name: str = "labels") -> None:
    """Validate that labels are integer and shape-compatible with ref."""
    assert labels.dtype in (torch.int64, torch.long), f"{name} must be int64/long, got {labels.dtype}"
    assert labels.shape == ref.shape, f"{name} shape {tuple(labels.shape)} != ref {tuple(ref.shape)}"


def set_seed(seed: int = 0) -> None:
    """Set Python, PyTorch, and Pyro random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


set_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Done here")
@dataclass
class Config:
    """Configuration for the VAE and training loop."""
    esm_model: str = "facebook/esm2_t12_35M_UR50D"
    gpt2_model: str = "nferruz/ProtGPT2"
    z_dim: int = 30
    prefix_len: int = 4
    adapter_rank: int = 128
    lr: float = 2e-4
    batch_size: int = 16
    epochs: int = 20
    kl_warmup_epochs: int = 10
    kl_start: float = 0.0
    kl_end: float = 1.0
    freeze_encoder: bool = True
    freeze_decoder: bool = True
    max_len: int = 50


CFG = Config()



# ---------------------------------------------------------------------------
# Pretrained models and tokenizers
# ---------------------------------------------------------------------------

def load_models_and_tokenizers(cfg: Config):
    """
    Load ESM-2 and ProtGPT2 with their tokenizers.

    ESM-2 is used as a frozen encoder (we take last_hidden_state).
    ProtGPT2 is used as a frozen decoder LM.
    """
    # ESM-2 encoder
    esm_tok = AutoTokenizer.from_pretrained(cfg.esm_model)
    esm_enc = EsmModel.from_pretrained(cfg.esm_model)
    if cfg.freeze_encoder:
        for p in esm_enc.parameters():
            p.requires_grad = False
    esm_enc.eval().to(DEVICE)
    esm_hidden = esm_enc.config.hidden_size

    # ProtGPT2 decoder
    prot_tok = GPT2TokenizerFast.from_pretrained(cfg.gpt2_model)
    prot_tok.pad_token = prot_tok.eos_token  # required for padding in batches, end of sentence token, indicates when to stop attention
    prot_lm = GPT2LMHeadModel.from_pretrained(
        cfg.gpt2_model,
        use_safetensors=True,
        #dtype="auto",
    )
    if cfg.freeze_decoder:
        for p in prot_lm.parameters():
            p.requires_grad = False
    prot_lm.eval().to(DEVICE)
    gpt2_hidden = prot_lm.config.n_embd

    # Basic consistency checks
    assert esm_hidden == esm_enc.config.hidden_size
    assert gpt2_hidden == prot_lm.config.n_embd
    assert_lastdim(prot_lm.transformer.wte.weight, gpt2_hidden, "ProtGPT2 embedding weight")

    return esm_tok, esm_enc, prot_tok, prot_lm, esm_hidden, gpt2_hidden


ESM_TOK, ESM_ENC, PROT_TOK, PROT_LM, ESM_HID, GPT2_HID = load_models_and_tokenizers(CFG)
GPT2_EMB = PROT_LM.transformer.wte  # embedding layer reused for teacher forcing


# ---------------------------------------------------------------------------
# Dataset and collate function
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    """
    Simple dataset of protein sequences.

    Each sequence is:
      - converted to uppercase
      - stripped of whitespace
      - truncated to max_len characters
    """
    def __init__(self, seqs: List[str], max_len: int):
        self.data: List[str] = []
        for s in seqs:
            if not s:
                continue
            t = str(s).strip().upper()
            if not t:
                continue
            self.data.append(t[:max_len])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


def collate_batch(batch: List[str]) -> Dict[str, torch.Tensor]:
    """
    Tokenize a batch of sequences for both ESM-2 and ProtGPT2.

    Returns a dictionary with:
      enc_input_ids, enc_attn_mask : for ESM-2
      dec_input_ids, dec_attn_mask : for ProtGPT2
    """
    esm_inputs = ESM_TOK(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CFG.max_len,
    )
    dec_inputs = PROT_TOK(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CFG.max_len,
    )
    assert esm_inputs.input_ids.ndim == 2
    assert dec_inputs.input_ids.ndim == 2
    return {
        "enc_input_ids": esm_inputs.input_ids,
        "enc_attn_mask": esm_inputs.attention_mask,
        "dec_input_ids": dec_inputs.input_ids,
        "dec_attn_mask": dec_inputs.attention_mask.bool(),
    }



# ---------------------------------------------------------------------------
# Small trainable components
# ---------------------------------------------------------------------------

class ESMToLatent(nn.Module):
    """
    Map ESM-2 sequence embeddings to Gaussian latent parameters (mu, logvar).

    Steps:
      - Masked mean over sequence length.
      - Two-layer MLP to produce concatenated mu and logvar (size 2*z_dim).
    """
    def __init__(self, esm_hidden: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 2 * z_dim),
        )

    def forward(self, enc_hidden: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_hidden: (B, T, H), attn_mask: (B, T)
        B, T, H = enc_hidden.shape
        assert H == ESM_HID
        assert_shape(attn_mask, B, T, name="ESM attn_mask")

        m = attn_mask.unsqueeze(-1).float()
        pooled = (enc_hidden * m).sum(1) / m.sum(1).clamp_min(1.0)   # (B, H)
        assert_lastdim(pooled, ESM_HID, "ESM pooled")

        out = self.net(pooled)                                       # (B, 2*z_dim)
        mu, logvar = out.chunk(2, dim=-1)
        assert_lastdim(mu, CFG.z_dim, "mu")
        assert_lastdim(logvar, CFG.z_dim, "logvar")
        return mu, logvar


class ZToPrefixAdapter(nn.Module):
    """
    Project latent z into a learned soft prefix for ProtGPT2.

    The adapter is a small two-layer MLP:
      z (B, z_dim) -> rank (B, rank) -> (B, prefix_len * gpt2_hidden)
      reshaped into (B, prefix_len, gpt2_hidden).
    """
    def __init__(self, z_dim: int, gpt2_hidden: int, prefix_len: int, rank: int = 128):
        super().__init__()
        self.gpt2_hidden = gpt2_hidden
        self.prefix_len = prefix_len
        self.proj1 = nn.Linear(z_dim, rank)
        self.act = nn.Tanh()
        self.proj2 = nn.Linear(rank, prefix_len * gpt2_hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        flat = self.proj2(self.act(self.proj1(z)))                   # (B, prefix_len * H)
        assert_lastdim(flat, self.prefix_len * self.gpt2_hidden, "adapter flat")
        prefix = flat.view(B, self.prefix_len, self.gpt2_hidden).contiguous()
        assert_shape(prefix, B, self.prefix_len, self.gpt2_hidden, name="prefix")
        return prefix


# ---------------------------------------------------------------------------
# VAE module
# ---------------------------------------------------------------------------

class VAE_ZPrefix(nn.Module):
    """
    Variational autoencoder for protein sequences with:

      - Encoder:   ESM-2 (frozen) + ESMToLatent head -> q(z | x)
      - Prior:     p(z) = Normal(0, I)
      - Decoder:   z -> soft prefix -> ProtGPT2 (frozen) -> p(x | z)

    Latent site:
      z is sampled under plate("batch_z", B, dim=-1) in both model and guide.

    Token likelihood:
      The categorical distribution over tokens has batch shape (B, L).
      A boolean mask (B, L) specifies which tokens contribute to the loss.
    """
    def __init__(self, z_dim: int, prefix_len: int):
        super().__init__()
        self.z_dim = z_dim
        self.prefix_len = prefix_len
        self.encoder_head = ESMToLatent(ESM_HID, z_dim)
        self.adapter = ZToPrefixAdapter(z_dim, GPT2_HID, prefix_len, rank=CFG.adapter_rank)
        self.kl_scale = 1.0

    # -------------------- model --------------------
    def model(self, batch: Dict[str, torch.Tensor]) -> None:
        pyro.module("vae", self)

        B = batch["dec_input_ids"].size(0)
        dec_ids  = batch["dec_input_ids"].to(DEVICE)   # (B, T_dec)
        dec_mask = batch["dec_attn_mask"].to(DEVICE)   # (B, T_dec)
        assert_same_device(dec_ids, dec_mask, next(PROT_LM.parameters()))

        # Latent z prior: independent across sequences
        with pyro.plate("batch_z", B, dim=-1):
            with poutine.scale(scale=self.kl_scale):
                base_prior = dist.Normal(
                    torch.zeros(self.z_dim, device=DEVICE),
                    torch.ones (self.z_dim, device=DEVICE),
                ).to_event(1)                           # batch_shape=(), event_shape=(z_dim,)
                z = pyro.sample("z", base_prior)        # (B, z_dim)
        assert_shape(z, B, self.z_dim, name="z")

        # Decode z into a soft prefix for ProtGPT2
        prefix = self.adapter(z)                        # (B, prefix_len, GPT2_HID)
        assert_shape(prefix, B, self.prefix_len, GPT2_HID, name="prefix")

        # Teacher-forcing setup: predict x_t from prefix + x_{<t}
        inputs  = dec_ids[:, :-1]                       # (B, T-1)
        targets = dec_ids[:, 1:].clone()                # (B, T-1)
        mask_tf = dec_mask[:, 1:]                       # (B, T-1)
        assert_bool_mask(mask_tf, targets.bool(), "mask_tf")

        token_embeds = GPT2_EMB(inputs)                 # (B, T-1, GPT2_HID)
        assert_shape(token_embeds, B, inputs.size(1), GPT2_HID, name="token_embeds")

        full_embeds = torch.cat([prefix, token_embeds], dim=1)  # (B, L, GPT2_HID)
        L = full_embeds.size(1)
        assert_shape(full_embeds, B, self.prefix_len + inputs.size(1), GPT2_HID, name="full_embeds")

        logits = PROT_LM(inputs_embeds=full_embeds).logits      # (B, L, V)
        V = PROT_LM.config.vocab_size
        assert logits.ndim == 3 and logits.size(0) == B and logits.size(1) == L and logits.size(2) == V

        # Build labels and mask for the likelihood.
        # Labels must always be valid token IDs; the mask determines which tokens contribute.
        eos_id = PROT_TOK.eos_token_id
        pad_prefix = torch.full((B, self.prefix_len), eos_id, dtype=targets.dtype, device=targets.device)
        labels = torch.cat([pad_prefix, targets], dim=1)        # (B, L)

        valid_mask = torch.cat(
            [
                torch.zeros((B, self.prefix_len), dtype=torch.bool, device=mask_tf.device),
                mask_tf,
            ],
            dim=1,
        )                                                        # (B, L)
        assert_int_labels(labels, labels, "labels")
        assert_bool_mask(valid_mask, labels.bool(), "valid_mask")

        labels_safe = labels.clone()
        labels_safe[~valid_mask] = eos_id
        assert (labels_safe >= 0).all() and (labels_safe < V).all(), "labels out of range"

        # Masked categorical likelihood over tokens
        cat    = dist.Categorical(logits=logits)                 # batch_shape = (B, L)
        masked = dist.MaskedDistribution(cat, valid_mask)        # mask (B, L)

        with pyro.plate("batch", B, dim=-2):
            with pyro.plate("time", L, dim=-1):
                pyro.sample("x_obs", masked, obs=labels_safe)

    # -------------------- guide --------------------
    def guide(self, batch: Dict[str, torch.Tensor]) -> None:
        B = batch["dec_input_ids"].size(0)
        enc_ids  = batch["enc_input_ids"].to(DEVICE)   # (B, T_enc)
        enc_mask = batch["enc_attn_mask"].to(DEVICE)   # (B, T_enc)
        assert_same_device(enc_ids, enc_mask, next(ESM_ENC.parameters()))

        # ESM-2 forward pass (encoder frozen)
        with torch.no_grad() if CFG.freeze_encoder else torch.enable_grad():
            enc_out = ESM_ENC(input_ids=enc_ids, attention_mask=enc_mask)
            hidden = enc_out.last_hidden_state                      # (B, T_enc, ESM_HID)

        mu, logvar = self.encoder_head(hidden, enc_mask)            # (B, z_dim), (B, z_dim)
        std = torch.exp(0.5 * logvar)

        # Latent z under the same plate and dimension as in the model
        with pyro.plate("batch_z", B, dim=-1):
            pyro.sample("z", dist.Normal(mu, std).to_event(1))

    # -------------------- generation --------------------
    @torch.no_grad()
    def generate(
        self,
        num: int = 3,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
    ) -> List[str]:
        """
        Sample sequences from the generative model.

        Steps:
          1. Draw z ~ N(0, I).
          2. Map z to a soft prefix (B, prefix_len, GPT2_HID).
          3. Call ProtGPT2.generate with this prefix as inputs_embeds and an
             explicit attention mask of ones.
          4. Decode the resulting token IDs to amino-acid sequences.

        Args
        ----
        num : number of sequences to sample.
        max_new_tokens : maximum number of tokens generated after the prefix.
        temperature : sampling temperature for the language model.

        Returns
        -------
        List[str]: decoded protein-like sequences.
        """
        z = torch.randn(num, self.z_dim, device=DEVICE)            # (num, z_dim)
        prefix = self.adapter(z)                                   # (num, prefix_len, GPT2_HID)
        B, Lp, H = prefix.shape
        assert_shape(prefix, B, self.prefix_len, GPT2_HID, name="prefix(gen)")

        # All positions in the soft prefix are valid context tokens.
        attn = torch.ones(B, Lp, dtype=torch.long, device=prefix.device)

        gen_ids = PROT_LM.generate(
            inputs_embeds=prefix,
            attention_mask=attn,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=PROT_TOK.eos_token_id,
            eos_token_id=PROT_TOK.eos_token_id,
        )

        sequences: List[str] = []
        for g in gen_ids:
            seq = PROT_TOK.decode(g, skip_special_tokens=True)
            sequences.append(seq)
        return sequences


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def kl_factor(epoch: int, start: float, end: float, warm: int) -> float:
    """
    Linear KL warmup schedule.

    Returns a factor in [start, end] that increases linearly over the first
    'warm' epochs and stays at 'end' afterwards.
    """
    if epoch >= warm:
        return end
    return start + (end - start) * float(epoch) / max(1, warm)


def train_epoch(loader: DataLoader, vae: VAE_ZPrefix, svi: SVI) -> Tuple[float, float]:
    """
    Train for a single epoch.

    Returns:
      nll_per_token : average negative log-likelihood per target token.
      ppl           : perplexity (exp of NLL per token).
    """
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        assert_same_device(batch["dec_input_ids"], next(PROT_LM.parameters()))

        # Number of next-token targets per sequence is (T - 1), summed over batch.
        tokens = (batch["dec_attn_mask"].sum(1) - 1).clamp_min(0).sum().item()
        loss = svi.step(batch)
        total_loss += float(loss)
        total_tokens += int(tokens)

    nll = total_loss / max(1, total_tokens)
    ppl = math.exp(nll)
    return nll, ppl


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Construct a small synthetic dataset by repeating a few toy sequences.
    toy_sequences = [
        "MKTFFVLVALLSSLSGQA",
        "MKAILVVLLYTLLPGAA",
        "GHHQVEGLGDNVALLR",
        "GHHQVEGIGDNVALLK",
        "GHHHVEGLGDNVALIR",
    ] * 200

    print("Here")

    dataset = ProteinDataset(toy_sequences, CFG.max_len)
    loader  = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    print("Done")

    # Inspect one batch to verify core shapes.
    first = next(iter(loader))
    print("ESM ids:", tuple(first["enc_input_ids"].shape), "Prot ids:", tuple(first["dec_input_ids"].shape))

    vae = VAE_ZPrefix(CFG.z_dim, CFG.prefix_len).to(DEVICE)
    pyro.clear_param_store()
    svi = SVI(vae.model, vae.guide, ClippedAdam({"lr": CFG.lr}), loss=Trace_ELBO())

    print("Starting tiny training run...")
    for epoch in range(1, CFG.epochs + 1):
        vae.kl_scale = kl_factor(epoch, CFG.kl_start, CFG.kl_end, CFG.kl_warmup_epochs)
        nll, ppl = train_epoch(loader, vae, svi)
        print(f"Epoch {epoch:02d} | KL scale {vae.kl_scale:.3f} | NLL/token {nll:.3f} | PPL {ppl:.2f}")

    print("\nSampling sequences:")
    samples = vae.generate(num=3, max_new_tokens=64, temperature=0.9)
    for i, seq in enumerate(samples):
        print(f"Sample {i+1}: {seq}")

