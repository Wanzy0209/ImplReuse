import tensorflow as tf

# Define the user-defined class similar to the PyTorch example
class Config:
    def __repr__(self):
        return "Config()"

# Define the computation function
# Note: batch_parallel passes inputs as a list of tensors to the computation function
def forward(inputs):
    # inputs is a list of tensors
    x = inputs[0]
    
    # Calling repr() on non-constant user object
    # This logic mirrors the PyTorch bug report: x * len(repr(config))
    # 'config' is lexically captured from the outer scope
    return x * len(repr(config))

# Initialize the user object and tensor
config = Config()
x = tf.random.normal((2, 2))

# Attempt to use the similar API: tf.compat.v1.tpu.batch_parallel
# This API compiles/traces the computation for parallel execution on TPUs.
try:
    # Note: Actual execution requires a TPU runtime. 
    # This test verifies the graph construction/tracing behavior with the repr() call.
    result = tf.compat.v1.tpu.batch_parallel(
        computation=forward,
        inputs=[x]
    )
    print("Graph construction successful. API accepted the repr() call.")
except Exception as e:
    # If TensorFlow has a similar tracing restriction for repr(), it would fail here.
    print(f"Error during tracing/compilation: {e}")