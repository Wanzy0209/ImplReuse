import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

def test_replication_pad3d_cache_consistency():
    """
    Test case for Issue 166012: tlparse entries on cache hit and not are inconsistent.
    
    This test uses torch.nn.ReplicationPad3d (Similar API) inside torch.compile (Original API).
    It replaces the 'tlparse' and 'torch._functorch.config' checks with 
    tf.raw_ops.Fingerprint to verify consistency between cache hits and misses.
    """
    
    # 1. Setup Model using Similar API: torch.nn.ReplicationPad3d
    padding = (1, 1, 1, 1, 1, 1)
    model = nn.ReplicationPad3d(padding)
    
    # 2. Prepare Input
    input_tensor = torch.randn(1, 3, 5, 5, 5)
    
    # 3. Replacement for torch._functorch.config_15.txt
    # Instead of checking configuration files, we fingerprint the input 
    # to ensure the test environment is stable and deterministic.
    input_np = input_tensor.numpy()
    input_tf = tf.constant(input_np)
    input_fingerprint = tf.raw_ops.Fingerprint(data=input_tf, method="farmhash64")
    print(f"Input Fingerprint (Config replacement): {input_fingerprint}")

    # 4. Compile using Original API: torch.compile
    compiled_model = torch.compile(model)
    
    # 5. Execution: Cache Miss
    output_miss = compiled_model(input_tensor)
    
    # 6. Execution: Cache Hit
    output_hit = compiled_model(input_tensor)
    
    # 7. Replacement for tlparse
    # Instead of parsing log entries (tlparse) to check for inconsistencies,
    # we compute the fingerprint of the outputs directly.
    
    # Convert PyTorch outputs to TensorFlow tensors for fingerprinting
    out_miss_np = output_miss.detach().numpy()
    out_hit_np = output_hit.detach().numpy()
    
    out_miss_tf = tf.constant(out_miss_np)
    out_hit_tf = tf.constant(out_hit_np)
    
    fp_miss = tf.raw_ops.Fingerprint(data=out_miss_tf, method="farmhash64")
    fp_hit = tf.raw_ops.Fingerprint(data=out_hit_tf, method="farmhash64")
    
    print(f"Cache Miss Fingerprint (tlparse replacement): {fp_miss}")
    print(f"Cache Hit Fingerprint (tlparse replacement):  {fp_hit}")
    
    # 8. Assertion
    # The bug report implies inconsistency. A valid test for the fix/behavior
    # asserts that the outputs (and thus their fingerprints) are identical.
    assert fp_miss == fp_hit, (
        f"Inconsistency detected between cache hit and miss!\n"
        f"Miss Fingerprint: {fp_miss}\n"
        f"Hit Fingerprint:  {fp_hit}"
    )
    
    print("Test Passed: Cache hit and miss outputs are consistent.")

if __name__ == "__main__":
    test_replication_pad3d_cache_consistency()