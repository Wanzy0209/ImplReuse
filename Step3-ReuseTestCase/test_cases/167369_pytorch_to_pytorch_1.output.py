import torch
import functools


class Config:
    def __repr__(self):
        return "Config()"


def forward(x, config):
    # Adapted to use functools.reduce instead of repr
    # This tests if Dynamo can trace the reduce builtin with user-defined objects
    configs = [config, config, config]
    
    # Use reduce to sum the lengths of the repr strings of the configs
    total_len = functools.reduce(lambda acc, c: acc + len(repr(c)), configs, 0)
    
    return x * total_len


config = Config()
x = torch.randn(2, 2)

compiled = torch.compile(forward, fullgraph=True)
result = compiled(x, config)

# Basic assertion to verify execution
assert result.shape == (2, 2)