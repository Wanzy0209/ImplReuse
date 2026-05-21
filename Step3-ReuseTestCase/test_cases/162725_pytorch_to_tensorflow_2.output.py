import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to ensure compatibility with v1 APIs like raw_rnn
# Note: raw_rnn is designed for graph mode. In TF 2.x, we wrap it in tf.function.
# We will compare standard Graph execution vs XLA (JIT) execution to mimic the 
# "eager vs compiled" consistency check in the original bug report.

def create_raw_rnn_fn(cell):
    def fn(inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Define loop_fn for raw_rnn
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                # Initialization (time = 0)
                next_input = inputs[:, 0, :]
                initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                emit_output = cell_output # None
                next_loop_state = loop_state
                finished = (time_steps <= 0)
            else:
                # Transition (time > 0)
                # Check if we have reached the end of the sequence
                finished = (time >= time_steps)
                
                if not finished:
                    next_input = inputs[:, time, :]
                else:
                    # Zero padding for finished steps
                    next_input = tf.zeros([batch_size, tf.shape(inputs)[2]], dtype=tf.float32)
                
                initial_state = cell_state
                emit_output = cell_output
                next_loop_state = loop_state
            
            return (finished, next_input, initial_state, emit_output, next_loop_state)

        # Call raw_rnn
        # raw_rnn returns (outputs, final_state)
        # outputs is a TensorArray
        outputs_ta, final_state = tf.compat.v1.nn.raw_rnn(cell, loop_fn)
        # Stack outputs to get a tensor of shape [time, batch, units]
        return outputs_ta.stack(), final_state
    return fn

# Setup inputs (mimicking sample_inputs)
batch_size = 2
time_steps = 5
input_dim = 4
hidden_dim = 8

# Create random inputs
inputs = tf.random.normal([batch_size, time_steps, input_dim], seed=42)

# Create RNN Cell
cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim)

# Get the function
rnn_fn = create_raw_rnn_fn(cell)

# 1. Standard Graph Execution (Equivalent to Eager in terms of baseline)
compiled_graph = tf.function(rnn_fn)
res1, state1 = compiled_graph(inputs)

# 2. XLA (JIT) Execution (Equivalent to torch.compile with inductor)
compiled_xla = tf.function(rnn_fn, jit_compile=True)
res2, state2 = compiled_xla(inputs)

# Assert results are close
# We check the output tensors
try:
    tf.debugging.assert_near(res1, res2, message="Outputs differ between Graph and XLA")
    print("Test Passed: Graph and XLA outputs are consistent.")
except tf.errors.InvalidArgumentError as e:
    print(f"Test Failed: {e}")