import torch
import tensorflow as tf

# Reproduce the environment of the fuzzed program
# Original PyTorch seed: 751735337
tf.random.set_seed(751735337)

def test_repeat_elements_divergence():
    # Recreating var_node_8: size=(13, 27), dtype=int16, value=3
    var_node_8 = tf.fill((13, 27), tf.constant(3, dtype=tf.int16))

    # Recreating index generation: torch.randint(0, 13, (11,), dtype=int64)
    # PyTorch randint high is exclusive.
    _input_size_var_node_7 = tf.shape(var_node_8)[0]
    _index_var_node_7 = tf.random.uniform(
        (11,),
        minval=0,
        maxval=_input_size_var_node_7,
        dtype=tf.int64
    )

    # Recreating var_node_7: torch.index_select(var_node_8, 0, _index_var_node_7)
    # TF equivalent is tf.gather
    var_node_7 = tf.gather(var_node_8, _index_var_node_7, axis=0)

    # Recreating var_node_6: torch.clamp(var_node_7, min=-1.0, max=1.0)
    var_node_6 = tf.clip_by_value(var_node_7, clip_value_min=-1.0, clip_value_max=1.0)

    # --- Testing the Similar API: tf.keras.backend.repeat_elements ---
    # We apply this to the tensor resulting from the dynamic operations.
    # This checks if repeat_elements handles the int16 dtype and dynamic shapes correctly.

    # 1. Eager Execution
    result_eager = tf.keras.backend.repeat_elements(var_node_6, rep=3, axis=1)

    # 2. Compiled Execution (Graph Mode)
    @tf.function
    def compiled_repeat(x):
        return tf.keras.backend.repeat_elements(x, rep=3, axis=1)

    result_compiled = compiled_repeat(var_node_6)

    # 3. Assertion: Check for divergence
    # The original bug was an assert failure due to divergence.
    # We verify that eager and compiled results match.
    assert tf.reduce_all(tf.equal(result_eager, result_compiled)).numpy(), \
        "Divergence detected between eager and compiled execution!"

    print("Test passed: No divergence between eager and compiled modes.")

if __name__ == "__main__":
    test_repeat_elements_divergence()