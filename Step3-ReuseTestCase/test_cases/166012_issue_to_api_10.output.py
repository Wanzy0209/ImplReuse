import torch
import tensorflow as tf
from tensorflow.experimental import dtensor

def test_layout_entry_consistency():
    """
    Test that the entries (sharding specs) of a dtensor.Layout are consistent
    across multiple accesses. This mirrors the expectation that cache entries
    (logs/files) in torch.compile should be consistent between a cache miss
    and a cache hit.
    """
    # Setup a mesh using CPU devices to ensure the test is runnable without TPU.
    # This simulates the distributed environment context.
    mesh = dtensor.Mesh(
        ["CPU:0", "CPU:1", "CPU:2", "CPU:3"],
        [("x", 2), ("y", 2)]
    )

    # Create a Layout with specific sharding specs.
    # These specs are the "entries" of the Layout object.
    layout = dtensor.Layout([dtensor.UNSHARDED, "x"], mesh)

    # --- Simulate Cache Miss (First Access) ---
    # Capture the state of the entries on the first inspection.
    # In the PyTorch bug, this corresponds to the list of files generated on a miss.
    specs_on_miss = list(layout.sharding_specs)
    repr_on_miss = repr(layout)
    
    # Verify the content of the entries on miss
    assert len(specs_on_miss) == 2, "Expected 2 sharding specs"
    assert specs_on_miss[0] == dtensor.UNSHARDED
    assert specs_on_miss[1] == "x"

    # --- Simulate Cache Hit (Second Access) ---
    # Capture the state of the entries on the second inspection.
    # In the PyTorch bug, this corresponds to the list of files (or lack thereof) on a hit.
    # We expect the Layout object to remain consistent and parsable.
    specs_on_hit = list(layout.sharding_specs)
    repr_on_hit = repr(layout)

    # --- Assertions for Consistency ---
    # 1. The list of entries (sharding specs) must be identical.
    # This prevents issues similar to the PyTorch bug where a parser (tlparse)
    # might encounter a different set of entries.
    assert specs_on_miss == specs_on_hit, \
        f"Inconsistent sharding specs between accesses: {specs_on_miss} vs {specs_on_hit}"

    # 2. The string representation must be identical.
    # This ensures that any logging or parsing mechanism relying on __repr__
    # behaves consistently.
    assert repr_on_miss == repr_on_hit, \
        f"Inconsistent layout repr between accesses: {repr_on_miss} vs {repr_on_hit}"

    print("Test passed: Layout entries are consistent across accesses.")

if __name__ == "__main__":
    test_layout_entry_consistency()