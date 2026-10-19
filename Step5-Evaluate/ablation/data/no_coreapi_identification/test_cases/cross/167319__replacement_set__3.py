import tensorflow as tf
from omegaconf import OmegaConf
import os

def test_tf_summary_value_omegaconf_serialization():
    """
    Test case to verify that tf.compat.v1.Summary.Value handles OmegaConf 
    objects correctly without causing unexpected serialization bloat or failures.
    
    This test mirrors the logic of the PyTorch bug (Issue 167319) where passing
    an OmegaConf ListConfig to an optimizer parameter led to the entire config
    tree being serialized. Here we check if passing an OmegaConf object to
    tf.compat.v1.Summary.Value causes similar issues.
    """
    
    # Create a config with nested structure (similar to the original bug report)
    cfg = OmegaConf.create({
        'model': {'betas': [0.9, 0.999]},
        'data': {'batch_size': 32},
        'other': {'secret': 'this_should_not_be_serialized'}
    })

    # Create a Summary.Value using the OmegaConf ListConfig
    # In the original bug, AdamW accepted the ListConfig directly.
    # Here, we pass it to the 'tensor' field or similar to see if it gets serialized.
    # Note: tf.compat.v1.Summary.Value expects specific types, but we are testing 
    # the behavior if a complex object is passed.
    
    # We use a simple value for the tag, but try to pass the complex config object
    # as the payload to see if it triggers the bloat.
    # Since Summary.Value is strict, we might need to wrap it or check the resulting proto.
    
    # To simulate the "Optimizer" context in TF, we might look at how optimizers handle configs,
    # but the Similar API is Summary.Value. Let's test Summary.Value directly.
    
    # Attempt to create a summary value. 
    # If we pass a list, it might be converted to a tensor.
    # The concern is if the OmegaConf parent reference is kept.
    
    # Create a tensor from the config list (this is what might happen internally)
    # If the conversion keeps the reference, the serialized summary will be huge.
    import numpy as np
    
    # Simulate passing the config data
    betas_data = cfg.model.betas
    
    # Create a Summary.Value
    # We use 'simple_value' for a scalar or 'tensor' for a tensor.
    # Let's assume we are logging a tensor.
    tensor_proto = tf.make_tensor_proto(betas_data)
    
    value = tf.compat.v1.Summary.Value(
        tag="test_betas",
        tensor=tensor_proto
    )
    
    # Serialize the summary to a string
    summary_str = tf.compat.v1.Summary(values=[value]).SerializeToString()
    
    # Check the size. 
    # If the bug exists (OmegaConf parent ref serialized), the size will be large 
    # (containing 'other': {'secret': ...}).
    # If the bug is fixed/absent, the size will be small (just the list [0.9, 0.999]).
    
    # A small threshold (e.g., 1KB) is enough to distinguish between just the list 
    # and the whole config tree.
    MAX_EXPECTED_SIZE = 1024 
    
    assert len(summary_str) < MAX_EXPECTED_SIZE, \
        f"Serialization bloat detected! Size: {len(summary_str)}. " \
        "The OmegaConf parent reference might have been serialized."

    # Clean up
    if os.path.exists('checkpoint.tf'):
        os.remove('checkpoint.tf')

if __name__ == "__main__":
    test_tf_summary_value_omegaconf_serialization()
    print("Test passed: No unexpected serialization bloat detected.")