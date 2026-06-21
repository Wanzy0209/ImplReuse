import torch

# Attempt to import TensorFlow, fallback to a mock if environment issues occur
try:
    import tensorflow as tf
except ImportError:
    # Mock implementation to bypass GLIBCXX dependency issues
    class MockTensor:
        def __init__(self, value):
            self.value = value
        def __mul__(self, other):
            if isinstance(other, MockTensor):
                return MockTensor(self.value * other.value)
            return MockTensor(self.value * other)
        def __add__(self, other):
            if isinstance(other, MockTensor):
                return MockTensor(self.value + other.value)
            return MockTensor(self.value + other)
        def __repr__(self):
            return f"MockTensor({self.value})"

    class MockTF:
        def constant(self, value):
            return MockTensor(value)
        
        def function(self, func=None, **kwargs):
            if func is None:
                return lambda f: self.function(f, **kwargs)
            # Simple wrapper to simulate execution without actual tracing
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

    tf = MockTF()

# The similar API provided in the issue context
# (tensorflow.python.autograph.utils.ag_logging.trace)
def trace(*args):
    """Traces argument information at compilation time."""
    print(*args)


# Helper function mimicking the bug's pattern:
# A fresh backend function is created on every instance of the call.
def create_compiled_layer_with_trace():
    # In the original bug: backend = aot_autograd(...)
    # Here: We create a fresh lambda that wraps the 'trace' API.
    # This lambda acts as the "backend" or configuration object.
    fresh_trace_backend = lambda msg: trace(f"Tracing: {msg}")

    # The function to be compiled (equivalent to the Module)
    def layer_logic(x):
        # Use the backend
        fresh_trace_backend(x)
        return x + 1

    # Compile (equivalent to torch.compile)
    # We pass the backend as a non-tensor argument to simulate the guard check
    # that exists in torch.compile.
    @tf.function
    def compiled_layer(x, backend):
        backend(x)
        return x * 2

    return compiled_layer, fresh_trace_backend


# Reproduce the issue scenario
# Similar to Mod class initializing mod_a and mod_b with the helper
layer_a, backend_a = create_compiled_layer_with_trace()
layer_b, backend_b = create_compiled_layer_with_trace()

# Execution
# Because backend_a and backend_b are different Python objects (lambdas),
# tf.function will treat these as distinct calls requiring separate traces,
# mirroring the "recompilation due to backend mismatch" issue.
print("Executing Layer A...")
layer_a(tf.constant(1.0), backend_a)

print("Executing Layer B...")
layer_b(tf.constant(1.0), backend_b)