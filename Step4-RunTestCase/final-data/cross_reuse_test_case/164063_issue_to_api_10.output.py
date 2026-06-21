import sys
import tempfile
import shutil

# Attempt to import dependencies with error handling for environment issues
try:
    import torch
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    # Handle environment incompatibility (e.g., GLIBCXX version mismatch)
    # which prevents TensorFlow from loading.
    print(f"Skipping test due to environment incompatibility or missing dependencies: {e}")
    sys.exit(0)

# This test case translates the PyTorch bug reproduction logic (variance on bfloat16)
# into the TensorFlow pattern using tf.saved_model.save (the Similar API).
# The goal is to verify the behavior of variance operations on bfloat16
# across eager execution and the saved graph model, checking for type consistency
# and numerical stability similar to the divergence reported in the PyTorch issue.

class VarianceModule(tf.Module):
    """
    A TensorFlow module that mimics the logic of the PyTorch function 'foo',
    specifically focusing on the reshape and variance operations that triggered the bug.
    """
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[36, 7112, 1, 1], dtype=tf.bfloat16)
    ])
    def compute(self, x):
        # PyTorch: t1 = t0.reshape((28, 24, 3, 127))
        # PyTorch: t2 = t1.var(dim=2)
        # Note: PyTorch var on bfloat16 often promotes to float32 internally.
        # TensorFlow's reduce_variance also promotes to float32 for bfloat16 inputs.
        reshaped = tf.reshape(x, (28, 24, 3, 127))
        variance = tf.math.reduce_variance(reshaped, axis=2)
        return variance

def test_variance_bfloat16_saved_model():
    # 1. Setup Input
    # Mimicking arg0 from the PyTorch issue: size=(36, 7112, 1, 1), dtype=bfloat16
    input_tensor = tf.random.uniform((36, 7112, 1, 1), dtype=tf.bfloat16)

    # 2. Instantiate Model
    model = VarianceModule()

    # 3. Eager Execution
    # In PyTorch, this corresponds to the first run of foo(...)
    eager_output = model.compute(input_tensor)

    # 4. Save the Model (Leveraging the Similar API: tf.saved_model.save)
    # This forces the graph to be traced and saved, similar to torch.compile
    export_dir = tempfile.mkdtemp()
    try:
        # Get the concrete function to ensure signature is fixed
        concrete_func = model.compute.get_concrete_function(input_tensor)
        
        tf.saved_model.save(
            model,
            export_dir,
            signatures=concrete_func
        )

        # 5. Load and Run (Graph Mode)
        loaded = tf.saved_model.load(export_dir)
        infer = loaded.signatures["serving_default"]
        graph_output = infer(input_tensor)

        # Extract output tensor from the dictionary returned by the loaded signature
        output_key = list(graph_output.keys())[0]
        graph_output_tensor = graph_output[output_key]

        # 6. Assertions
        # The PyTorch bug was a TypeError('unexpected type fp32'), implying a type mismatch.
        # We check if the dtype is consistent between Eager and Graph modes.
        assert eager_output.dtype == graph_output_tensor.dtype, \
            f"Dtype mismatch: Eager {eager_output.dtype} vs Graph {graph_output_tensor.dtype}"
        
        # Check for numerical divergence
        np.testing.assert_allclose(
            eager_output.numpy(), 
            graph_output_tensor.numpy(), 
            rtol=1e-5, 
            atol=1e-5,
            err_msg="Output mismatch between Eager and SavedModel execution"
        )
        
        print("Test Passed: Eager and SavedModel (Graph) execution are consistent.")

    finally:
        # Cleanup
        shutil.rmtree(export_dir, ignore_errors=True)

if __name__ == "__main__":
    test_variance_bfloat16_saved_model()