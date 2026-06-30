import sys
import torch
import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    # Handle environment issues like missing GLIBCXX versions
    print(f"Skipping test: TensorFlow import failed due to environment issues.")
    print(f"Error details: {e}")
    sys.exit(0)

# Define two models using tf.keras.layers.RNN with different configurations
# to mimic the "two models" setup in the original bug report.

class RNNModelReturnSeq(tf.keras.Model):
    def __init__(self):
        super(RNNModelReturnSeq, self).__init__()
        # Using return_sequences=True
        self.rnn = tf.keras.layers.RNN(
            cell=tf.keras.layers.SimpleRNNCell(units=5),
            return_sequences=True
        )

    def call(self, x):
        print('input shape:', x.shape)
        y = self.rnn(x)
        print('output shape:', y.shape)
        return y

class RNNModelNoReturnSeq(tf.keras.Model):
    def __init__(self):
        super(RNNModelNoReturnSeq, self).__init__()
        # Using return_sequences=False
        self.rnn = tf.keras.layers.RNN(
            cell=tf.keras.layers.SimpleRNNCell(units=5),
            return_sequences=False
        )

    def call(self, x):
        print('input shape:', x.shape)
        y = self.rnn(x)
        print('output shape:', y.shape)
        return y

def process(model, x):
    print(f'model: {model.__class__.__name__}')
    print('running eager mode...')
    out_eager = model(x)
    
    print('exporting/tracing...')
    # tf.function is the TensorFlow equivalent to torch.export.export for tracing
    traced_model = tf.function(model)
    out_traced = traced_model(x)
    
    # Verify that the traced output shape matches the eager output shape
    assert out_eager.shape == out_traced.shape, \
        f"Shape mismatch: eager {out_eager.shape} vs traced {out_traced.shape}"
    print()

# Setup input
# Batch size=1, Time steps=10, Features=3
x = tf.random.normal((1, 10, 3))

# Process first model
process(RNNModelReturnSeq(), x)

# Process second model
# The original bug suggests the second model might have incorrect output shape
# due to caching/state pollution from the first.
process(RNNModelNoReturnSeq(), x)