import torch
import torchvision
import tensorflow as tf
import numpy as np

# Disable eager execution to use tf.compat.v1.metrics
tf.compat.v1.disable_eager_execution()

# 1. Model setup
model = torchvision.models.mobilenet_v2(weights=None)

# 2. Input generation
# Original: x = torch.rand((1, 3, 224, 224))
# Replacement: tf.nn.relu
# To make this runnable and semantically similar (generating input),
# we generate a random tensor and apply ReLU.
tf_x = tf.nn.relu(tf.random.uniform((1, 3, 224, 224)))
x = torch.from_numpy(tf_x.numpy())

# 3. Export
ep = torch.export.export(model, (x,))

# 4. Inference
res1 = model(x)
res2 = ep.module()(x)

# 5. Assertion
# Original: torch.testing.assert_close(res1, res2)
# Replacement: tf.compat.v1.metrics.true_positives
# We interpret the assertion as checking if the top-1 predictions match.
# Convert PyTorch tensors to TensorFlow tensors
labels = tf.constant(np.argmax(res1.detach().numpy(), axis=1))
predictions = tf.constant(np.argmax(res2.detach().numpy(), axis=1))

# Setup the metric
tp, update_op = tf.compat.v1.metrics.true_positives(labels, predictions)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.local_variables_initializer())
    sess.run(update_op)
    tp_val = sess.run(tp)
    # If the models are identical, all predictions should be true positives (1 out of 1).
    assert tp_val == 1, f"Test failed: True Positives = {tp_val}, expected 1"