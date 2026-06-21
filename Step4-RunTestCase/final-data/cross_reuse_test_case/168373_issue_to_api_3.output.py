import tensorflow as tf
from tensorflow.errors import UnimplementedError

# This test case translates the PyTorch module structure from the original bug report
# into TensorFlow. It preserves the logic of defining sub-modules and a main module,
# applying compilation (tf.function), and executing the graph.
# It leverages the similar API (tf.errors.UnimplementedError) to handle potential
# backend compilation failures, which is the TensorFlow equivalent of encountering
# an unsupported operation during the compilation process.

class SubMod(tf.Module):
    def __init__(self):
        super().__init__()

    # Replaced jit_compile with experimental_compile for compatibility with older TensorFlow versions
    @tf.function(experimental_compile=True)
    def forward(self, x):
        return tf.sin(x)

class Mod(tf.Module):
    def __init__(self):
        super().__init__()
        self.mod_a = SubMod()
        self.mod_b = SubMod()

    # Replaced jit_compile with experimental_compile for compatibility with older TensorFlow versions
    @tf.function(experimental_compile=True)
    def forward(self, x):
        return self.mod_a.forward(x) + self.mod_b.forward(x)

def test_module_compilation():
    mod = Mod()
    x = tf.random.normal((4,))

    # Verify that the compiled module executes without raising
    # an UnimplementedError, which would indicate a backend mismatch
    # or unsupported operation in the compiled context.
    try:
        result = mod.forward(x)
        assert result.shape == (4,)
        print("Test Passed: Module compiled and executed successfully.")
    except UnimplementedError as e:
        print(f"Test Failed: Encountered UnimplementedError during compilation/execution - {e}")
        raise

if __name__ == "__main__":
    test_module_compilation()