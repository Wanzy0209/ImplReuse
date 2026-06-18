import torch
import tensorflow as tf
import numpy as np

def test_initializer_serialization_precision():
    """
    Test case adapted from PyTorch issue #161944.
    
    Original Issue: torch.compile (Inductor) uses fast math for exp, leading to 
    precision loss compared to eager execution or float64 reference.
    
    Adaptation: Check if tf.keras.initializers.serialize (and deserialize) 
    preserves the precision and behavior of the initializer, analogous to 
    checking if compilation preserves the precision of the function.
    """
    
    # Setup: Define a shape and an initializer
    # Analogous to: inp = torch.randn(8192)
    shape = (8192,)
    
    # Analogous to: func = torch.exp
    # We use an initializer with specific float parameters to test precision retention
    mean_val = 0.123456789
    stddev_val = 0.987654321
    initializer = tf.keras.initializers.RandomNormal(mean=mean_val, stddev=stddev_val)

    # 1. Eager execution (Reference)
    # Analogous to: out1 = func(inp)
    tf.random.set_seed(42)
    out1 = initializer(shape)

    # 2. Transformed execution (Serialized -> Deserialized)
    # Analogous to: out2 = torch.compile(func)(inp)
    # Here, the "transformation" is the serialization process.
    config = tf.keras.initializers.serialize(initializer)
    restored_initializer = tf.keras.initializers.deserialize(config)
    
    tf.random.set_seed(42)
    out2 = restored_initializer(shape)

    # 3. High Precision Reference (Original Object)
    # In the context of serialization, the original object acts as the high-precision reference.
    # We compare the output of the restored object against the original.
    
    # Calculate max absolute difference
    # Analogous to: print((out3_high - out2).abs().max())
    diff = tf.reduce_max(tf.abs(out1 - out2))
    
    print(f"Max difference between original and serialized/deserialized: {diff.numpy()}")
    
    # Assert that the serialization process did not alter the behavior/precision
    # Note: For random initializers, with the same seed, outputs should be identical.
    assert diff < 1e-6, f"Serialization introduced precision drift: {diff}"

    # Additionally check parameter preservation (Configuration precision)
    assert restored_initializer.mean == mean_val, "Mean parameter precision lost"
    assert restored_initializer.stddev == stddev_val, "Stddev parameter precision lost"

if __name__ == "__main__":
    test_initializer_serialization_precision()