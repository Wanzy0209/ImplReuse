import torch
import torch.nn.functional as F
import tensorflow as tf

# Original: class Config: ...
# To use tf.keras.optimizers.serialize, we need a Keras Optimizer instance.
# We use SGD as a minimal example.
config = tf.keras.optimizers.SGD()

# Original: x = torch.randn(2, 2)
# Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
# We generate inputs for the loss function to produce a tensor of shape (2, 2)
input_tensor = torch.randn(2, 2)
target_tensor = torch.ones(2, 2)
x = F.hinge_embedding_loss(input_tensor, target_tensor, reduction='none')

def forward(x, config):
    # Original: return x * len(repr(config))
    # Replacement: repr -> tf.keras.optimizers.serialize
    # tf.keras.optimizers.serialize returns a dict, len() gets the number of keys.
    return x * len(tf.keras.optimizers.serialize(config))

# Original: compiled = torch.compile(forward, fullgraph=True)
compiled = torch.compile(forward, fullgraph=True)

# Execute the compiled function
# Note: This reproduces the logic of calling a representation/serialization function
# inside a compiled graph, similar to the original repr() bug.
try:
    result = compiled(x, config)
    print("Test passed. Result:", result)
except Exception as e:
    print("Test failed with error:", e)