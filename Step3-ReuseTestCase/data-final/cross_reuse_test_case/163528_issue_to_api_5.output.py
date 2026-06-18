import torch
import torch.nn as nn
import sys

def test_searchsorted_compile_with_cropping1d_pattern():
    """
    Test case for torch.searchsorted with torch.compile on GPU.
    This test leverages the data structure and dimension handling pattern
    from tf.keras.layers.Cropping1D (temporal sequence on axis 1) to verify
    the correctness of searchsorted compilation.
    """
    
    # Leverage the input shape from the tf.keras.layers.Cropping1D example
    # input_shape = (2, 3, 2) -> (Batch, Time/Sequence, Features)
    batch_size = 2
    time_steps = 3
    features = 2
    
    # Create a module that mimics the usage pattern
    class SearchSortedModule(nn.Module):
        def __init__(self, boundaries):
            super().__init__()
            # Register boundaries as a buffer (non-trainable parameter)
            self.register_buffer('boundaries', boundaries)

        def forward(self, x):
            # Perform searchsorted along the time dimension (dim=1), 
            # similar to how Cropping1D operates on axis 1.
            return torch.searchsorted(self.boundaries, x, dim=1)

    # Generate data similar to the Cropping1D example: np.arange(...).reshape(...)
    # We create sorted boundaries for valid searchsorted behavior
    values = torch.arange(batch_size * time_steps * features, dtype=torch.float32).reshape(batch_size, time_steps, features)
    
    # Create boundaries (sorted along dim 1)
    boundaries = torch.arange(batch_size * time_steps * features, dtype=torch.float32).reshape(batch_size, time_steps, features)
    boundaries = torch.sort(boundaries, dim=1)[0]

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    else:
        print("CUDA not available, skipping GPU test.", file=sys.stderr)

    for device in devices:
        print(f"\nTesting on device: {device}")
        
        # Move data to device
        v_dev = values.to(device)
        b_dev = boundaries.to(device)
        
        # Initialize models
        model_eager = SearchSortedModule(b_dev).to(device)
        model_compiled = torch.compile(SearchSortedModule(b_dev).to(device), fullgraph=True)

        # Warmup
        with torch.no_grad():
            _ = model_eager(v_dev)
            _ = model_compiled(v_dev)

        # Inference
        with torch.no_grad():
            res_eager = model_eager(v_dev)
            res_compiled = model_compiled(v_dev)

        # Check for differences
        diff = torch.max(torch.abs(res_eager - res_compiled)).item()
        print(f"  Max difference between eager and compiled: {diff}")
        
        # Assert results are identical
        try:
            assert torch.equal(res_eager, res_compiled), \
                f"Results differ on {device}! Eager:\n{res_eager}\nCompiled:\n{res_compiled}"
            print(f"  Test PASSED on {device}")
        except AssertionError as e:
            print(f"  Test FAILED on {device}: {e}")
            raise

if __name__ == "__main__":
    test_searchsorted_compile_with_cropping1d_pattern()