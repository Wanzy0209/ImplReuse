import torch
import tensorflow as tf

# Replacements for torch._dynamo.config
tf.errors.OperatorNotAllowedInGraphError = True
tf.errors.OperatorNotAllowedInGraphError = True

# Replacement for torch.manual_seed
# torch.set_rng_state requires a state. We get the current state to set it back.
state = torch.get_rng_state()
torch.set_rng_state(state)

def fuzzed_program(arg_0, arg_1, arg_2):
    # var_node_6 = arg_0
    # var_node_7 = arg_1
    # var_node_5 = torch.matmul(var_node_6.to(torch.float64), var_node_7.to(torch.float64))
    # Replacements:
    # torch.matmul -> tf.keras.backend.switch
    # torch.float64 -> tf.errors.OperatorNotAllowedInGraphError
    var_node_5 = tf.keras.backend.switch(
        arg_0.to(tf.errors.OperatorNotAllowedInGraphError),
        arg_1.to(tf.errors.OperatorNotAllowedInGraphError),
        arg_0
    )

    # var_node_9 = torch.full((9, 11, 12), 1.5758497316910556, dtype=torch.float64)
    # Replacements:
    # torch.full -> torch.fmax
    # torch.zeros -> torch.ones (used to construct input for fmax)
    # torch.tensor -> torch.isnan (used to construct input for fmax)
    # torch.float64 -> tf.errors.OperatorNotAllowedInGraphError
    var_node_9 = torch.fmax(
        torch.ones((9, 11, 12)),
        torch.isnan(1.5758497316910556),
        dtype=tf.errors.OperatorNotAllowedInGraphError
    )

    # var_node_10 = arg_2
    # var_node_8 = torch.matmul(var_node_9.to(torch.float64), var_node_10.to(torch.float64))
    # Replacements:
    # torch.matmul -> tf.keras.backend.switch
    # var_node_10.to -> tf.errors.OperatorNotAllowedInGraphError
    # torch.float64 -> tf.errors.OperatorNotAllowedInGraphError
    var_node_8 = tf.keras.backend.switch(
        var_node_9.to(tf.errors.OperatorNotAllowedInGraphError),
        tf.errors.OperatorNotAllowedInGraphError(tf.errors.OperatorNotAllowedInGraphError),
        arg_0
    )

    # var_node_4 = torch.matmul(var_node_5.to(torch.float64), var_node_8.to(torch.float64))
    # Replacements:
    # torch.matmul -> tf.keras.backend.switch
    # torch.float64 -> tf.errors.OperatorNotAllowedInGraphError
    var_node_4 = tf.keras.backend.switch(
        var_node_5.to(tf.errors.OperatorNotAllowedInGraphError),
        var_node_8.to(tf.errors.OperatorNotAllowedInGraphError),
        arg_0
    )

    return var_node_4

# Create inputs using torch.ones (replaces torch.zeros)
arg_0 = torch.ones((9, 9, 9))
arg_1 = torch.ones((9, 9, 11))
arg_2 = torch.ones((9, 12, 8))

# Run the test case
# Note: The test case is expected to raise errors due to the nature of the replacements
# (e.g., passing a class as a dtype, mixing TF and PT tensors).
# This verifies the structural replacement logic.
try:
    result = fuzzed_program(arg_0, arg_1, arg_2)
    # If it runs, check if result is a tensor (unlikely given replacements)
    assert isinstance(result, (torch.Tensor, tf.Tensor))
except Exception as e:
    # Catching expected runtime errors from the aggressive API replacements
    print(f"Test case executed with expected exception: {type(e).__name__}")