import torch
import pytest
from torch.optim import AdamW
from omegaconf import OmegaConf

# Attempt to import the similar API. If not available, we mock it to ensure the test is runnable.
# This leverages the similar API (tf.compat.v1.tpu.PaddingSpec) as a candidate for reuse
# in the context of type validation, similar to how OmegaConf ListConfig is handled.
try:
    from tensorflow.python.tpu.tpu import PaddingSpec
except ImportError:
    from enum import IntEnum
    # Mocking PaddingSpec if TensorFlow is not installed to ensure test portability
    class PaddingSpec(IntEnum):
        AUTO = 0
        POWER_OF_TWO = 1

def test_adamw_betas_type_validation():
    """
    Test that AdamW validates the 'betas' parameter type.
    It should reject OmegaConf ListConfig and other non-tuple/list sequence-like objects
    (such as the similar API PaddingSpec) to prevent unexpected serialization behavior.
    """
    model = torch.nn.Linear(10, 1)

    # 1. Test with OmegaConf ListConfig (Original Bug)
    # This reproduces the original issue where ListConfig was accepted.
    cfg = OmegaConf.create({
        'model': {'betas': [0.9, 0.999]},
        'data': {'batch_size': 32}
    })
    
    # Expected behavior: Raise TypeError
    with pytest.raises(TypeError, match="betas must be a tuple or list"):
        optimizer = AdamW(model.parameters(), lr=1e-3, betas=cfg.model.betas)

    # 2. Test with Similar API (PaddingSpec)
    # PaddingSpec is an IntEnum. While iterable, it is not a tuple or list of floats.
    # Leveraging this API tests the robustness of the type validation against other
    # sequence-like objects that might carry unexpected state or types.
    
    # Passing the Enum class itself (iterable over members)
    with pytest.raises(TypeError, match="betas must be a tuple or list"):
        optimizer = AdamW(model.parameters(), lr=1e-3, betas=PaddingSpec)

    # Passing a list containing Enum members (wrong content type)
    with pytest.raises(TypeError, match="betas must be a tuple or list"):
        optimizer = AdamW(model.parameters(), lr=1e-3, betas=[PaddingSpec.AUTO, PaddingSpec.POWER_OF_TWO])

    # 3. Test valid input to ensure the optimizer still works correctly
    # This confirms we didn't break the valid path.
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    assert optimizer is not None