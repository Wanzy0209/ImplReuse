import tensorflow as tf

# Setup: Define parameters for the distribution (analogous to tokenizer inputs)
example_params = (0.0, 1.0)

# Setup: Create the distribution (analogous to loading the model)
# We use Normal distribution which is typically fully reparameterized
dist = tf.compat.v1.distributions.Normal(loc=example_params[0], scale=example_params[1])

# Action: Access the reparameterization_type property
# The original bug involved an assertion error during export regarding active modes.
# Here we verify that the reparameterization type is correctly registered and accessible.
reparam_type = dist.reparameterization_type

# Assertion: Verify the behavior matches the expected constant
assert reparam_type == tf.compat.v1.distributions.FULLY_REPARAMETERIZED

# Additional verification: Ensure the property is accessible in graph mode (TF's export context)
@tf.function
def check_in_graph_mode():
    d = tf.compat.v1.distributions.Normal(loc=0.0, scale=1.0)
    return d.reparameterization_type

graph_reparam_type = check_in_graph_mode()
assert graph_reparam_type == tf.compat.v1.distributions.FULLY_REPARAMETERIZED