import pytest
import torch
from torch.optim import AdamW
from omegaconf import OmegaConf, ListConfig

def test_adamw_betas_type_validation():
    """
    Test that AdamW validates the 'betas' parameter type.
    
    This test ensures that passing an OmegaConf ListConfig (or other non-list/tuple 
    sequence-like objects) raises a TypeError, preventing unexpected serialization 
    behavior where the entire configuration tree might be retained.
    
    This relates to the pattern seen in configuration APIs like tf.io.VarLenFeature,
    where strict type adherence is crucial for correct serialization and parsing behavior.
    """
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    
    # Create a config with nested structure using OmegaConf
    # This mimics the usage of configuration objects like VarLenFeature in TF pipelines
    cfg = OmegaConf.create({
        'optimizer': {
            'lr': 1e-3,
            'betas': [0.9, 0.999]
        },
        'other_nested_config': {'data': 'value'}
    })

    # Extract the betas as a ListConfig (not a list or tuple)
    betas_config = cfg.optimizer.betas
    assert isinstance(betas_config, ListConfig), "Setup failed: betas should be a ListConfig"

    # Test 1: Valid inputs (list and tuple) should work
    try:
        AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        AdamW(model.parameters(), lr=1e-3, betas=[0.9, 0.999])
    except Exception as e:
        pytest.fail(f"AdamW raised an unexpected exception with valid betas types: {e}")

    # Test 2: OmegaConf ListConfig should raise TypeError
    # This prevents the serialization bug where the parent config tree is saved
    with pytest.raises(TypeError) as excinfo:
        AdamW(model.parameters(), lr=1e-3, betas=betas_config)
    
    # Verify the error message mentions the expected types
    assert "betas" in str(excinfo.value).lower()
    assert "tuple" in str(excinfo.value) or "list" in str(excinfo.value)

if __name__ == "__main__":
    test_adamw_betas_type_validation()
    print("Test passed successfully.")