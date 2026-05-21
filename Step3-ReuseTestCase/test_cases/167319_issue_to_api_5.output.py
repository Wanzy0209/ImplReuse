import torch
import pytest
from torch.optim import AdamW
from omegaconf import OmegaConf
import torch.profiler.itt as itt

def test_adamw_betas_type_validation():
    """
    Test that AdamW validates the 'betas' parameter type to prevent 
    unexpected serialization behavior with OmegaConf objects.
    Leverages torch.profiler.itt.range_pop to scope the test execution.
    """
    # Create a config with nested structure (Reproduces the bug setup)
    cfg = OmegaConf.create({
        'model': {'betas': [0.9, 0.999]},
        'data': {'batch_size': 32},
    })

    model = torch.nn.Linear(10, 1)

    # Leverage the similar API (torch.profiler.itt.range_push/pop) 
    # to profile the critical section of the test.
    itt.range_push("test_optimizer_validation")

    # Expected behavior: Optimizer should validate that betas is a tuple or list
    # and raise a TypeError if other sequence-like objects (like ListConfig) are passed.
    with pytest.raises(TypeError):
        optimizer = AdamW(model.parameters(), lr=1e-3, betas=cfg.model.betas)

    # Verify that valid types (tuple) still work correctly
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    assert optimizer is not None

    # Verify that valid types (list) still work correctly
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=[0.9, 0.999])
    assert optimizer is not None

    itt.range_pop()