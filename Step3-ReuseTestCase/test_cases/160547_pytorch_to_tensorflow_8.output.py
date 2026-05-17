import torch
import tensorflow as tf
from collections import namedtuple

def test_rnn_namedtuple():
    """
    Adapted test case for tf.keras.layers.RNN based on PyTorch issue 160547.
    Verifies if the RNN layer can handle namedtuple inputs, particularly 
    when traced (exported) via tf.function.
    """
    # Define a namedtuple to hold inputs (sequence and initial state)
    RNNInputs = namedtuple('RNNInputs', ['sequence', 'state'])

    # Define a Model wrapping tf.keras.layers.RNN
    class RNNModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # Using a SimpleRNNCell inside the RNN layer
            self.rnn = tf.keras.layers.RNN(
                cell=tf.keras.layers.SimpleRNNCell(units=4),
                return_sequences=False,
                return_state=False
            )

        def call(self, inputs):
            # Unpack the namedtuple similar to the PyTorch example
            x, y = inputs.sequence, inputs.state
            # Pass unpacked arguments to the RNN layer
            return self.rnn(x, initial_state=y)

    # Prepare inputs
    # Batch size=3, Timesteps=5, Features=4, Units=4
    batch_size = 3
    timesteps = 5
    features = 4
    units = 4

    inp = RNNInputs(
        sequence=tf.ones((batch_size, timesteps, features)),
        state=tf.zeros((batch_size, units))
    )

    model = RNNModel()

    # 1. Test direct call (Eager execution)
    # Analogous to print(M()(*inp)) in the original bug report
    print("Testing direct call (Eager)...")
    try:
        output_eager = model(inp)
        print(f"Direct call succeeded. Output shape: {output_eager.shape}")
    except Exception as e:
        print(f"Direct call failed: {e}")

    # 2. Test traced call (tf.function)
    # Analogous to torch.export.export(M(), inp, strict=False)
    # In TensorFlow, tf.function is the mechanism for graph capture/tracing.
    print("\nTesting traced call (tf.function)...")
    try:
        # Converting the model to a tf.function graph
        exported_model = tf.function(model)
        output_traced = exported_model(inp)
        print(f"Traced call succeeded. Output shape: {output_traced.shape}")
        
        # Verify consistency between eager and traced execution
        assert tf.reduce_all(tf.abs(output_eager - output_traced) < 1e-6).numpy()
        print("Outputs match between eager and traced execution.")
    except Exception as e:
        print(f"Traced call failed: {e}")

    # 3. Test workaround (Standard tuple)
    # Analogous to the workaround converting namedtuple to kwargs/tuple
    print("\nTesting workaround (standard tuple)...")
    try:
        # Convert namedtuple to a standard tuple
        inp_tuple = (inp.sequence, inp.state)
        output_tuple = model(inp_tuple)
        print(f"Tuple call succeeded. Output shape: {output_tuple.shape}")
    except Exception as e:
        print(f"Tuple call failed: {e}")

if __name__ == "__main__":
    test_rnn_namedtuple()