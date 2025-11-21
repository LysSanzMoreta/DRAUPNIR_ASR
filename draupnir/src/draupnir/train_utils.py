import torch.optim
from pyro.optim.optim import PyroOptim
from torch import Tensor
from typing import Any, Dict, Iterable, List, Optional, Union, ValuesView

class NoamOpt():
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor=2, warmup=4000, optimizer=torch.optim.AdamW):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))



class NoamOpt2():
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor=2, warmup=4000, optimizer=torch.optim.AdamW):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    # def step(self):
    #     "Update parameters and rate"
    #     self._step += 1
    #     rate = self.rate()
    #     for p in self.optimizer.param_groups:
    #         p['lr'] = rate
    #     self._rate = rate
    #     self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"

        if step is None:
            step = self._step

        step = max(step, 1)

        # rate = self.factor * \
        #     (self.model_size ** (-0.5) *
        #      min(step ** (-0.5), step * self.warmup ** (-1.5)))
        #
        # print(rate)

        rate = (self.model_size ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))

        return rate





