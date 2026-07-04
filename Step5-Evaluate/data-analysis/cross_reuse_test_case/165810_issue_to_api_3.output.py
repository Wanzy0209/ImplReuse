import tensorflow as tf
import sys

# The original PyTorch issue (165810) describes a bug where stack traces
# (metadata) on graph nodes become incorrect or lost after a graph
# transformation (aot_export_joint_with_descriptors) due to regenerate_from_base.
#
# This test targets the similar API: tf.mlir.experimental.run_pass_pipeline.
# We verify that location information (analogous to stack traces) is
# preserved correctly when running a transformation pipeline, ensuring
# the transformation does not corrupt or drop metadata.

def test_run_pass_pipeline_preserves_metadata():
    # Try to locate the run_pass_pipeline function.
    # It might be in tf.mlir.experimental or tf.compiler.mlir.experimental
    # depending on the TensorFlow version.
    run_pass_pipeline = None
    try:
        # Try the standard location for MLIR tools in TensorFlow
        if hasattr(tf, 'compiler') and hasattr(tf.compiler, 'mlir') and hasattr(tf.compiler.mlir, 'experimental'):
            run_pass_pipeline = tf.compiler.mlir.experimental.run_pass_pipeline
        # Fallback to the location in the original code if available
        elif hasattr(tf, 'mlir') and hasattr(tf.mlir, 'experimental'):
            run_pass_pipeline = tf.mlir.experimental.run_pass_pipeline
    except AttributeError:
        pass

    if run_pass_pipeline is None:
        print("Skipping test: tf.mlir.experimental.run_pass_pipeline (or tf.compiler.mlir.experimental) not found in this TensorFlow version.")
        return

    # Define an MLIR module with explicit location information.
    # This simulates a graph with stack trace annotations attached to nodes.
    mlir_input = """
    module {
      func.func @test_function(%arg0: tensor<i32>) -> tensor<i32> {
        %0 = "tf.Add"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32> loc("user_script.py:42")
        return %0 : tensor<i32>
      }
    }
    """

    # Define a pass pipeline. The canonicalizer performs standard optimizations
    # and rewrites, similar to how PyTorch's regeneration might transform nodes.
    pass_pipeline = "canonicalizer"

    # Case 1: Run with show_debug_info=True.
    # We expect the location information to be present in the output.
    output_with_debug = run_pass_pipeline(
        mlir_input, pass_pipeline, show_debug_info=True
    )

    # Assert that the specific location is preserved.
    # If the location was lost or wrong (as in the PyTorch bug), this would fail.
    assert "loc(\"user_script.py:42\")" in output_with_debug, \
        "Test failed: Location metadata was lost or incorrect during pass pipeline execution."

    # Case 2: Run with show_debug_info=False.
    # We expect the location information to be stripped.
    output_without_debug = run_pass_pipeline(
        mlir_input, pass_pipeline, show_debug_info=False
    )

    assert "loc(\"user_script.py:42\")" not in output_without_debug, \
        "Test failed: Location metadata should be stripped when show_debug_info is False."

    print("Test passed: Metadata handling in run_pass_pipeline is correct.")

if __name__ == "__main__":
    test_run_pass_pipeline_preserves_metadata()