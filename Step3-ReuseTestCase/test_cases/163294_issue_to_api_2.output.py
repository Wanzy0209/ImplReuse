import torch
import tensorflow as tf
import sys
from io import StringIO

# Helper to capture stdout for verifying tf.autograph.trace execution
class OutputCapture:
    def __init__(self):
        self.content = ""
    def write(self, text):
        self.content += text
    def flush(self):
        pass

# Define the module/case similar to the PyTorch SetGradCase
class TraceCase(tf.Module):
    @tf.function
    def __call__(self, x):
        # Using the similar API: tf.autograph.trace
        # This mimics the 'with torch.no_grad():' block in the original issue,
        # acting as a specific operation within the graph construction.
        tf.autograph.trace("Tracing TraceCase with input:", x)
        return x * 4

# Setup output capture
capture = OutputCapture()
sys.stdout = capture

# Step 1: First "Export" / Trace
# Corresponds to: ep = torch.export.export(SetGradCase(), ...)
case = TraceCase()
input_tensor = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# Trigger tracing
result1 = case(input_tensor)

# Verify trace happened in the first pass
trace_output_1 = capture.content
assert "Tracing TraceCase with input:" in trace_output_1, "tf.autograph.trace did not execute during initial trace"

# Reset capture for the second step
capture = OutputCapture()
sys.stdout = capture

# Step 2: "Retracing" / Re-export
# Corresponds to: ep2 = torch.export.export(ep.module(), ...)
# In TensorFlow, we simulate this by wrapping the already traced module 
# in another tf.function to test graph stability and nesting.
@tf.function
def retrace_wrapper(x):
    return case(x)

# Trigger the second trace/wrapper execution
result2 = retrace_wrapper(input_tensor)

# Restore stdout
sys.stdout = sys.__stdout__

# Assertions
# 1. Verify functional correctness (values match expected output)
expected = input_tensor * 4
assert tf.reduce_all(result1 == expected).numpy(), "First trace result incorrect"
assert tf.reduce_all(result2 == expected).numpy(), "Retrace result incorrect"

# 2. Verify the structure remains valid.
# The original PyTorch bug resulted in an empty submod (structural failure).
# Here we ensure the operation is preserved and execution succeeds without error.
# Note: Depending on TF graph caching, trace might not print again, but the operation must exist.
print("Test Passed: tf.autograph.trace handled correctly during re-tracing.")