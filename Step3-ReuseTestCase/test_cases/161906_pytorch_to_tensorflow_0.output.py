import torch
import tensorflow as tf

class M(tf.Module):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        # Introduce parameter for broadcasting / stride ops (mimicking original nn.Parameter)
        self.p = tf.Variable(2.0, dtype=tf.float32)

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        # Mimic the math operations from the original bug to ensure
        # we aren't just passing a raw tensor, but something derived.
        # Original: R = torch.abs(S.real); I = S.imag / self.p
        # Adapted: Apply scaling using the parameter
        x_scaled = x / self.p

        # Target API: tf.compat.v1.nn.dropout
        # Note: keep_prob = 1 - rate
        keep_prob = 1.0 - self.rate
        y = tf.compat.v1.nn.dropout(x_scaled, keep_prob=keep_prob, seed=0)
        return y

def main():
    # Setup device (TF handles device placement automatically or via context managers)
    tf.random.set_seed(0)

    # Input tensor (matching shape from original)
    x = tf.random.normal((1, 16000), dtype=tf.float32)
    m = M()

    # Eager: works fine
    print("Running Eager mode...")
    y_eager = m.forward(x)
    assert y_eager.shape == x.shape
    print("Eager mode OK:", y_eager.shape)

    # Compile: tf.function (Graph mode)
    # This is the TensorFlow equivalent of torch.compile
    print("Running Graph mode (tf.function)...")
    m_compiled = tf.function(m.forward)
    y_compiled = m_compiled(x)

    assert y_compiled.shape == x.shape
    print("Graph mode OK:", y_compiled.shape)

if __name__ == "__main__":
    main()