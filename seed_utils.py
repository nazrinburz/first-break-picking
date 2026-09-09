"""
One place to seed everything that has randomness in it: python's random
module, numpy, and torch (CPU + CUDA if it's around). Call this once at
the top of a training run, before building the model or any DataLoader.

Note this only gets you same-machine, same-package-version reproducibility
- exact bitwise reproducibility across different GPUs/driver versions isn't
something a seed alone guarantees, but for "rerun this and get basically
the same numbers" it's what you want.
"""
import random
import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed=0):
    """A torch.Generator tied to the seed, for DataLoader(shuffle=True,
    generator=...) and WeightedRandomSampler(generator=...) - without this,
    those two draw from torch's global RNG state, which set_seed() does
    seed, but passing an explicit generator makes the dependency obvious
    and keeps it from drifting if something else touches the global RNG
    in between (e.g. a stray random call while debugging in a notebook)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
