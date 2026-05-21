import torch
import tensorflow as tf
import tensorflow.experimental.dtensor as dtensor

def test_copy_to_mesh_strict_mode():
    """
    Adapts the logic of torch.compile(fullgraph=True) vs torch.compiler.disable
    to TensorFlow.
    
    In PyTorch:
    - torch.compile(fullgraph=True) enforces a strict graph (no breaks).
    - torch.compiler.disable attempts to exclude a function.
    - The bug was that strict mode rejected the exclusion.
    
    In TensorFlow:
    - tf.function(jit_compile=True) enforces strict XLA compilation.
    - tf.experimental.dtensor.copy_to_mesh handles data layout/movement.
    - This test verifies if copy_to_mesh works within a strict compilation context,
      similar to verifying if the 'disable' exclusion works in 'fullgraph' mode.
    """
    
    # Setup a basic mesh and layout for DTensor
    # Using CPU to ensure the test runs without GPU requirements
    mesh = dtensor.create_mesh(['batch'], ['device:CPU:0'])
    layout = dtensor.Layout([dtensor.UNSHARDED], mesh)
    
    # Define a function with strict compilation enabled (jit_compile=True)
    # This is analogous to torch.compile(fullgraph=True)
    @tf.function(jit_compile=True)
    def strict_compile_fn(tensor):
        # Call the API under test.
        # This is analogous to calling a torch.compiler.disable'd function
        # inside a fullgraph compile.
        return dtensor.copy_to_mesh(tensor, layout)

    # Create a regular tensor
    regular_tensor = tf.constant([1.0, 2.0, 3.0])
    
    # Execute the function
    # In the PyTorch bug, this raises torch._dynamo.exc.Unsupported.
    # Here we check if the TensorFlow equivalent handles the operation
    # successfully or raises a similar error.
    try:
        result = strict_compile_fn(regular_tensor)
        
        # Verify the result is a DTensor
        assert isinstance(result, dtensor.DTensor), "Result should be a DTensor"
        
        # Verify the values are preserved
        # Note: accessing .values might be necessary depending on DTensor version behavior
        # but usually direct comparison works if sharding allows.
        # For UNSHARDED, it should match.
        tf.debugging.assert_equal(result, tf.constant([1.0, 2.0, 3.0]))
        
        print("Test Passed: copy_to_mesh works within strict compilation context.")
        
    except Exception as e:
        # If this fails, it mimics the PyTorch bug behavior
        print(f"Test Failed: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    test_copy_to_mesh_strict_mode()