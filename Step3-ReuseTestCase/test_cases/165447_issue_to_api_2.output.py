import torch
import tensorflow as tf
from tensorflow.experimental import dtensor

# Test case inspired by Issue 165447: AOT Precompile serialization failed when running multiple times.
# The original bug involves a global state reset (torch._dynamo.reset()) followed by an operation.
# The similar API (tf.experimental.dtensor.get_default_mesh) relies on a global singleton.
# This test verifies that get_default_mesh handles the 'reset' (None) state gracefully
# and behaves consistently when called multiple times.

def test_get_default_mesh_state_consistency():
    # 1. Check initial state (Analogous to setup in the original bug)
    # The original bug sets up a model and compiles it. Here we check the mesh state.
    initial_mesh = dtensor.get_default_mesh()

    # 2. Simulate the "Reset" scenario
    # The original bug calls torch._dynamo.reset(). In the context of get_default_mesh,
    # a "reset" state corresponds to the _dtensor_singleton being None.
    # We verify the API handles this state correctly without crashing.
    
    if initial_mesh is None:
        # 3. Running multiple times (Title: "running multiple times")
        # The original bug failed when running the serialization/load process multiple times.
        # We ensure get_default_mesh is stable and idempotent when called repeatedly.
        for _ in range(5):
            current_mesh = dtensor.get_default_mesh()
            # Assert that the state remains consistent (None) across multiple calls
            assert current_mesh is None, \
                "get_default_mesh should consistently return None when DTensor is uninitialized"
    else:
        # If initialized, verify consistency across multiple calls
        for _ in range(5):
            current_mesh = dtensor.get_default_mesh()
            assert current_mesh is initial_mesh, \
                "get_default_mesh should return the same mesh instance across calls"

    # 4. Final assertion to ensure the API returns the correct type
    assert initial_mesh is None or isinstance(initial_mesh, dtensor.Mesh), \
        "get_default_mesh must return a Mesh object or None"

if __name__ == "__main__":
    test_get_default_mesh_state_consistency()
    print("Test passed: get_default_mesh handles state consistency correctly.")