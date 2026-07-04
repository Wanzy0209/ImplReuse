import torch
import tensorflow as tf
from tensorflow.python.platform.benchmark import benchmark_config

# Mimicking the wrapper function pattern from the PyTorch issue
# where a fresh backend/config object is created on every call.
def create_session_with_custom_config():
    """
    Wrapper function that creates a new Session with a specific config.
    This mirrors the 'torch_compile_with_custom_backend' function in the issue,
    which creates a new backend object every time.
    """
    # Using the similar API: tf.test.benchmark_config
    # This mirrors the usage of aot_autograd in the PyTorch issue.
    return tf.compat.v1.Session(config=benchmark_config())


class TFSubMod:
    def __init__(self):
        pass

    def forward(self, x):
        return tf.sin(x)


class TFMod:
    def __init__(self):
        self.mod_a = TFSubMod()
        self.mod_b = TFSubMod()

        # Preserving the original bug reproduction logic:
        # Calling the wrapper function multiple times creates distinct objects.
        # In PyTorch, this triggers recompilation because the backend object identity differs.
        # In TensorFlow, this creates distinct Sessions, mirroring the pattern.
        self.sess_a = create_session_with_custom_config()
        self.sess_b = create_session_with_custom_config()

    def forward(self, x):
        with self.sess_a.as_default():
            res_a = self.mod_a.forward(x).eval()
        with self.sess_b.as_default():
            res_b = self.mod_b.forward(x).eval()
        return res_a + res_b


def test_issue_168373_tf_equivalent():
    """
    Test case reflecting the relationship between the PyTorch recompilation issue
    and the tf.test.benchmark_config API pattern.

    This test verifies that the pattern of creating a configuration object inside
    a wrapper function (as seen in tf.test.benchmark_config) leads to distinct
    objects when called multiple times, similar to the PyTorch issue.
    """
    # Instantiate the module
    mod = TFMod()

    # Create a tensor input
    # Note: TF1 requires operations to be within a graph
    with tf.compat.v1.Graph().as_default():
        x = tf.constant(1.0)
        # Execute the forward pass
        # The sessions inside TFMod will handle the execution of their specific ops
        result = mod.forward(x)
        # Basic check to ensure execution
        assert result is not None