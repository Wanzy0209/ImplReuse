import torch
import torch.nn as nn


class Config:
    """
    A user-defined class similar to configuration objects used in quantization or serialization.
    """
    def __init__(self, name="default"):
        self.name = name

    def __repr__(self):
        # This repr implementation triggers the bug in Dynamo
        return f"Config(name='{self.name}')"


def forward(x, config):
    """
    Forward function that uses repr() on a user object.
    Adapted to return a tuple (data, metadata) similar to the pattern 
    found in tf.keras.quantizers.serialize (returning byte_str, version).
    """
    # Calling repr() on non-constant user object
    # This triggers the bug without the fix
    metadata = len(repr(config))
    
    # Perform computation based on the repr length
    result = x * metadata
    
    # Return a tuple to mimic the structure of the similar API
    return result, metadata


def test_compile_repr_with_tuple_return():
    """
    Test case to verify torch.compile handles repr() on user-defined objects
    correctly, even when the function returns a tuple structure similar to
    serialization APIs.
    """
    config = Config()
    x = torch.randn(2, 2)

    # Eager execution
    expected_tensor, expected_meta = forward(x, config)

    # Compiled execution
    compiled = torch.compile(forward, fullgraph=True)
    result_tensor, result_meta = compiled(x, config)

    # Assertions to verify correctness
    assert torch.allclose(expected_tensor, result_tensor), "Tensor output mismatch"
    assert expected_meta == result_meta, "Metadata output mismatch"

if __name__ == "__main__":
    test_compile_repr_with_tuple_return()
    print("Test passed.")