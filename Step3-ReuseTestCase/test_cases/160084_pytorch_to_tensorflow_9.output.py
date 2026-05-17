import torch
import tensorflow as tf

class RegressionModel(tf.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        # Adaptation: torch.nn.Parameter -> tf.Variable
        self.a = tf.Variable(float(a), dtype=tf.float32)
        self.b = tf.Variable(float(b), dtype=tf.float32)
        self.first_batch = True

    # Adaptation: torch.compile -> tf.function (graph compilation)
    @tf.function
    def __call__(self, x=None):
        # Adaptation: Using tf.name_scope as the target API
        with tf.name_scope("regression_forward"):
            if self.first_batch:
                # Adaptation: Python print -> tf.print for graph execution compatibility
                tf.print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
                self.first_batch.assign(False)
            return x * self.a + self.b

def test_regression_model_with_name_scope():
    """
    Test case adapted from PyTorch issue 160084.
    Verifies that a stateful model with side-effects (print/state mutation)
    runs correctly within a tf.name_scope and tf.function context.
    """
    model = RegressionModel()
    
    # Adaptation: torch.randn -> tf.random.normal
    inputs = tf.random.normal((4, 10))
    
    # First run (triggers print and state update)
    output = model(inputs)
    
    # Assertions
    assert output.shape == (4, 10), "Output shape mismatch"
    assert not model.first_batch, "State flag should be False after first batch"
    
    # Second run (verifies stability and no re-compilation issues)
    output2 = model(inputs)
    assert output2.shape == (4, 10), "Output shape mismatch on second run"

if __name__ == "__main__":
    test_regression_model_with_name_scope()
    print("Test passed successfully.")