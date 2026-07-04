import torch
import torch.nn as nn
import math

def generate_kaiser_like_input(length, dtype=torch.float32, device=None):
    """
    Mimics the logic found in tf.signal.kaiser_window to generate a tensor.
    Original TF logic:
      halflen_float = (cast(window_length) - 1.0) / 2.0
      arg = range(-halflen_float, halflen_float + 0.1)
    """
    halflen = (length - 1.0) / 2.0
    # Using torch.arange to replicate tf.range behavior
    return torch.arange(-halflen, halflen + 0.1, dtype=dtype, device=device)

class ExpandReproModel(nn.Module):
    """
    Model designed to reproduce the issue where torch.expand is interpreted 
    as torch.repeat by the compiler, leading to OOM.
    """
    def __init__(self, expand_factor):
        super().__init__()
        self.expand_factor = expand_factor

    def forward(self, x):
        # Loop structure similar to the original bug report
        results = []
        for i in range(x.size(1)):
            # 1. Generate input using logic similar to the Similar API (tf.signal.kaiser_window)
            # We treat the spatial dimension as the 'window_length'
            window = generate_kaiser_like_input(x.size(2), device=x.device)
            
            # 2. Apply the Original API Under Test (torch.expand)
            # This is the operation that triggers the bug in Inductor.
            # If interpreted as repeat, it allocates expand_factor * size memory.
            # If interpreted as expand (view), it allocates 0 extra memory.
            expanded = window.expand(self.expand_factor, -1)
            
            # Perform a reduction to ensure the operation is not optimized away
            results.append(expanded.sum())
        
        return torch.stack(results)

def test_expand_memory_efficiency():
    """
    Test case to verify that torch.expand is treated as a view (zero-copy)
    and not as a copy (repeat) inside a torch.compile graph.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Check for torch.compile availability (PyTorch 2.0+)
    if not hasattr(torch, "compile"):
        print("torch.compile is not available (requires PyTorch 2.0+), skipping test.")
        return

    # Configuration
    # A large expansion factor that would cause OOM if treated as repeat,
    # but fits easily if treated as a view.
    EXPAND_FACTOR = 10000 
    BASE_SIZE = 100
    BATCH_SIZE = 1
    CHANNELS = 3

    model = ExpandReproModel(EXPAND_FACTOR).cuda()
    
    # Compile with Inductor (the backend where the bug was reported)
    # The bug manifests in default, reduce-overhead, and max-autotune modes.
    model = torch.compile(model, mode="default")

    # Create dummy input
    x = torch.randn((BATCH_SIZE, CHANNELS, BASE_SIZE), device='cuda')

    try:
        # Run forward pass
        output = model(x)
        
        # If we reach here, the operation likely succeeded without OOM,
        # implying expand was treated as a view.
        assert output.shape == (CHANNELS,), f"Expected output shape ({CHANNELS},), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaNs"
        
        print("Test Passed: torch.expand handled efficiently as a view in compiled model.")

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda out of memory" in error_msg:
            raise AssertionError(
                "Bug Reproduced: torch.expand was likely interpreted as torch.repeat, "
                "causing an Out Of Memory error."
            ) from e
        else:
            # Re-raise other runtime errors
            raise

if __name__ == "__main__":
    test_expand_memory_efficiency()