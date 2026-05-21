import torch
import torch.nn as nn

def test_constantpad3d_compile_cache_consistency():
    """
    Test case to reproduce the logic of Issue 166012 using torch.nn.ConstantPad3d.
    
    The original issue describes inconsistent 'tlparse' entries (internal logging/metadata)
    between a cache miss and a cache hit when using torch.compile. This test sets up
    the scenario where a model containing the similar API (ConstantPad3d) is compiled
    and run twice to trigger both cache states.
    """
    # Define a model using the similar API: torch.nn.ConstantPad3d
    # Padding format: (left, right, top, bottom, front, back)
    model = nn.ConstantPad3d(padding=(1, 1, 2, 2, 0, 0), value=3.5)

    # Compile the model using the Original API Under Test: torch.compile
    # This enables the caching mechanism where the inconsistency was reported.
    compiled_model = torch.compile(model)

    # Create a dummy input tensor (Batch, Channel, Depth, Height, Width)
    input_tensor = torch.randn(2, 3, 5, 5, 5)

    # --- Step 1: Cache Miss ---
    # The first call triggers the compilation process (Dynamo, AOTAutograd, Inductor).
    # This generates the initial set of artifacts (graphs, logs, etc.).
    output_miss = compiled_model(input_tensor)

    # --- Step 2: Cache Hit ---
    # The second call with the same input shape attempts to reuse the compiled program.
    # The bug report indicates that internal logs (tlparse) might be inconsistent here.
    output_hit = compiled_model(input_tensor)

    # --- Verification ---
    # 1. Ensure functional consistency between cache miss and hit.
    assert torch.equal(output_miss, output_hit), \
        "Outputs differ between cache miss and cache hit."

    # 2. Ensure correctness against the eager (non-compiled) execution.
    expected_output = model(input_tensor)
    assert torch.allclose(output_hit, expected_output), \
        "Compiled output does not match eager output."

if __name__ == "__main__":
    test_constantpad3d_compile_cache_consistency()