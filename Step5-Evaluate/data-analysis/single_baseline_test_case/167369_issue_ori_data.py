# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn as nn


class Config:
    def __repr__(self):
        return "Config()"


def forward(x, config):
    # Calling repr() on non-constant user object
    # This triggers the bug without the fix
    return x * len(repr(config))


config = Config()
x = torch.randn(2, 2)

compiled = torch.compile(forward, fullgraph=True)