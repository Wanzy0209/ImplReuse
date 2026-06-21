import torch
import tensorflow as tf
import sys
import io

def test_tf_autograph_trace_conv_bias():
    """
    Test case leveraging tf.autograph.trace to verify tensor shapes during graph construction.
    
    This test mirrors the logic of the original PyTorch bug (Issue 166797), where 
    torch.onnx.export with dynamo produced an incorrect bias tensor shape for a Conv layer.
    Here, we use tf.autograph.trace to inspect the bias shape during the tf.function 
    tracing phase to ensure correctness.
    """
    
    # Capture stdout to verify the trace output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Define a model with a Convolutional layer (mimicking the ResNet50 layer in the bug)
    class SimpleConvModel(tf.Module):
        def __init__(self):
            super().__init__()
            # Initialize a Conv2D layer with bias
            # The bug report specifically mentioned incorrect bias tensor shape
            self.conv = tf.keras.layers.Conv2D(filters=64, kernel_size=7, use_bias=True)

        @tf.function
        def __call__(self, x):
            # Use tf.autograph.trace to inspect the bias tensor during the tracing phase
            # This corresponds to inspecting the ONNX graph nodes in the original issue
            tf.autograph.trace("Inspecting Bias Tensor Shape:", self.conv.bias.shape)
            return self.conv(x)

    # Create model and dummy input
    model = SimpleConvModel()
    # Input shape: (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, 224, 224, 3))

    # Execute the model to trigger tracing
    _ = model(dummy_input)

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Verify the output
    output = captured_output.getvalue()
    
    # Assertions
    # 1. Check that the trace was executed
    assert "Inspecting Bias Tensor Shape:" in output, "tf.autograph.trace did not execute."
    
    # 2. Check that the bias shape is correct (64,)
    # The original bug had an "obviously incorrect" shape. Here we assert the correct one.
    assert "(64,)" in output, f"Expected bias shape (64,) in trace output, got: {output}"
    
    print("Test passed: tf.autograph.trace correctly reported bias shape during graph construction.")

if __name__ == "__main__":
    test_tf_autograph_trace_conv_bias()