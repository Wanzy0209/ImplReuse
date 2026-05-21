import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
tf.random.set_seed(42)

# Define inputs for CTC Loss
# Batch size: 2, Max time: 4, Num classes: 5
batch_size = 2
max_time = 4
num_classes = 5

# Logits: [batch_size, max_time, num_classes]
logits = tf.random.normal([batch_size, max_time, num_classes], dtype=tf.float32)

# Labels: SparseTensor
# Example labels: [[0, 1], [1, 2]]
indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
values = [0, 1, 1, 2]
dense_shape = [batch_size, 2] # Max label length 2
labels = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

# Sequence length
sequence_length = [max_time, max_time]

# --- Eager Execution ---
with tf.GradientTape() as tape_eager:
    # Note: ctc_loss performs softmax internally, so inputs are logits
    loss_eager = tf.compat.v1.nn.ctc_loss(
        labels=labels,
        logits=logits,
        sequence_length=sequence_length,
        time_major=False # logits is [batch, time, classes]
    )
    # Reduce to scalar for gradient
    loss_eager_reduced = tf.reduce_mean(loss_eager)

eager_grad = tape_eager.gradient(loss_eager_reduced, logits)

# --- Graph Execution (tf.function) ---
# In TensorFlow, tf.function captures the computation graph similar to torch.cuda.graph
@tf.function
def graph_fn(logits_in, labels_in, seq_len_in):
    with tf.GradientTape() as tape:
        loss_graph = tf.compat.v1.nn.ctc_loss(
            labels=labels_in,
            logits=logits_in,
            sequence_length=seq_len_in,
            time_major=False
        )
        loss_graph_reduced = tf.reduce_mean(loss_graph)
    return tape.gradient(loss_graph_reduced, logits_in)

graph_grad = graph_fn(logits, labels, sequence_length)

# --- Verification ---
# Check if gradients match
assert np.allclose(eager_grad.numpy(), graph_grad.numpy(), rtol=1e-5, atol=1e-5), "Mismatch in gradient outputs"

print("Eager Gradient:\n", eager_grad.numpy())
print("Graph Gradient:\n", graph_grad.numpy())
print("Test Passed.")