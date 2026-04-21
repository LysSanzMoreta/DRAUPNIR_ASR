from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PottsShape:
    """Normalized shape information for Potts parameters."""

    batch_size: int
    length: int
    alphabet_size: int
    rank: int


class LowRankPottsPseudoLikelihood:
    """Low-rank Potts pseudo-likelihood for aligned integer-coded sequences.

    Parameters are accepted either in shared form
    - fields:  [L, A]
    - factors: [L, A, R]

    or in batched form
    - fields:  [B, L, A]
    - factors: [B, L, A, R].
    """

    def __init__(self, fields: torch.Tensor, factors: torch.Tensor, center_factors: bool = True) -> None:
        self.fields = fields
        self.factors = factors
        self.center_factors = bool(center_factors)

        self._validate_parameters()
        self._fields, self._factors, self.shape = self._normalize_parameters()

    def _validate_parameters(self) -> None:
        """Validate field and factor tensors before normalization."""
        assert isinstance(self.fields, torch.Tensor), "fields must be a tensor"
        assert isinstance(self.factors, torch.Tensor), "factors must be a tensor"
        assert self.fields.ndim in (2, 3), "fields must have shape [L, A] or [B, L, A]"
        assert self.factors.ndim in (3, 4), "factors must have shape [L, A, R] or [B, L, A, R]"
        assert self.fields.device == self.factors.device, "fields and factors must be on the same device"
        assert self.fields.dtype.is_floating_point, "fields must be floating point"
        assert self.factors.dtype.is_floating_point, "factors must be floating point"

        if self.fields.ndim == 2:
            length = self.fields.shape[0]
            alphabet_size = self.fields.shape[1]
            assert self.factors.ndim == 3, "unbatched fields require unbatched factors"
            assert self.factors.shape[0] == length, "length mismatch"
            assert self.factors.shape[1] == alphabet_size, "alphabet mismatch"
            assert self.factors.shape[2] >= 1, "rank must be positive"
        else:
            batch_size = self.fields.shape[0]
            length = self.fields.shape[1]
            alphabet_size = self.fields.shape[2]
            assert self.factors.ndim == 4, "batched fields require batched factors"
            assert self.factors.shape[0] == batch_size, "batch mismatch"
            assert self.factors.shape[1] == length, "length mismatch"
            assert self.factors.shape[2] == alphabet_size, "alphabet mismatch"
            assert self.factors.shape[3] >= 1, "rank must be positive"

    def _normalize_parameters(self) -> tuple[torch.Tensor, torch.Tensor, PottsShape]:
        """Convert parameters to batched form and optionally center factors."""
        if self.fields.ndim == 2:
            normalized_fields = self.fields.unsqueeze(0)
            normalized_factors = self.factors.unsqueeze(0)
        else:
            normalized_fields = self.fields
            normalized_factors = self.factors

        if self.center_factors:
            normalized_factors = normalized_factors - normalized_factors.mean(dim=2, keepdim=True)

        shape = PottsShape(
            batch_size=normalized_fields.shape[0],
            length=normalized_fields.shape[1],
            alphabet_size=normalized_fields.shape[2],
            rank=normalized_factors.shape[3],
        )
        return normalized_fields, normalized_factors, shape

    def _broadcast_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Broadcast parameters to the batch size of x."""
        assert isinstance(x, torch.Tensor), "x must be a tensor"
        assert x.ndim in (1, 2), "x must have shape [L] or [B, L]"
        assert x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long), "x must be integer-coded"
        assert x.device == self.fields.device, "x and parameters must be on the same device"

        squeeze_result = False
        if x.ndim == 1:
            x_batched = x.unsqueeze(0)
            squeeze_result = True
        else:
            x_batched = x

        assert x_batched.shape[1] == self.shape.length, "sequence length mismatch"
        assert torch.all(x_batched >= 0), "states must be non-negative"
        assert torch.all(x_batched < self.shape.alphabet_size), "states are out of range"

        broadcast_fields = self._fields
        broadcast_factors = self._factors

        if broadcast_fields.shape[0] == 1 and x_batched.shape[0] > 1:
            broadcast_fields = broadcast_fields.expand(x_batched.shape[0], -1, -1)
            broadcast_factors = broadcast_factors.expand(x_batched.shape[0], -1, -1, -1)
        else:
            assert broadcast_fields.shape[0] == x_batched.shape[0], "batch size mismatch between x and parameters"

        return x_batched.long(), broadcast_fields, broadcast_factors, squeeze_result

    def conditional_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute site-conditional logits for every residue state."""
        x_batched, fields, factors, squeeze_result = self._broadcast_inputs(x)

        batch_size = x_batched.shape[0]
        length = x_batched.shape[1]
        rank = factors.shape[3]

        gather_index = x_batched.unsqueeze(-1).unsqueeze(-1).expand(batch_size, length, 1, rank)
        selected_factors = torch.gather(factors, dim=2, index=gather_index).squeeze(2)
        assert selected_factors.shape == (batch_size, length, rank)

        total_sum = selected_factors.sum(dim=1, keepdim=True)
        context_sum = total_sum - selected_factors

        pairwise_term = (factors * context_sum.unsqueeze(2)).sum(dim=-1)
        logits = fields + pairwise_term

        assert logits.shape == (batch_size, length, self.shape.alphabet_size)
        if squeeze_result:
            return logits.squeeze(0)
        return logits

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pseudo-log-likelihood for each sequence."""
        x_batched, _, _, squeeze_result = self._broadcast_inputs(x)
        logits = self.conditional_logits(x_batched)
        log_probs = F.log_softmax(logits, dim=-1)
        observed = torch.gather(log_probs, dim=-1, index=x_batched.unsqueeze(-1)).squeeze(-1)
        total = observed.sum(dim=-1)

        assert total.shape == (x_batched.shape[0],)
        if squeeze_result:
            return total.squeeze(0)
        return total

    def sample_gibbs(self, num_sweeps: int = 10, initial_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Draw approximate Gibbs samples from the induced Potts model.

        This is a batch sampler. The returned tensor always has shape [B, L].
        """
        assert num_sweeps >= 1, "num_sweeps must be positive"

        batch_size = self.shape.batch_size
        length = self.shape.length
        rank = self.shape.rank
        fields = self._fields
        factors = self._factors

        if initial_x is None:
            x = torch.argmax(fields, dim=-1)
        else:
            assert initial_x.ndim == 2, "initial_x must have shape [B, L]"
            assert initial_x.shape == (batch_size, length), "initial_x has wrong shape"
            x = initial_x.clone().long()

        gather_index = x.unsqueeze(-1).unsqueeze(-1).expand(batch_size, length, 1, rank)
        selected = torch.gather(factors, dim=2, index=gather_index).squeeze(2)
        total_sum = selected.sum(dim=1)

        for _ in range(num_sweeps):
            for site in range(length):
                context = total_sum - selected[:, site, :]
                logits = fields[:, site, :] + (factors[:, site, :, :] * context.unsqueeze(1)).sum(dim=-1)
                distribution = torch.distributions.Categorical(logits=logits)
                new_state = distribution.sample()

                new_index = new_state.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, rank)
                new_vector = torch.gather(factors[:, site, :, :], dim=1, index=new_index).squeeze(1)

                x[:, site] = new_state
                selected[:, site, :] = new_vector
                total_sum = context + new_vector

        assert x.shape == (batch_size, length)
        return x