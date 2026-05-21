import torch
import tensorflow as tf

# Adaptation of the test case for tf.summary.experimental.get_step
# The original bug report highlights an "Eager/Compile Divergence".
# This test verifies that tf.summary.experimental.get_step behaves consistently
# in both Eager mode and Graph (tf.function) mode.

def get_step_wrapper():
    """
    Mimics the 'foo' function from the original test case.
    Instead of tensor operations, we retrieve the summary step.
    """
    # Retrieve the step set for the current thread
    step = tf.summary.experimental.get_step()
    return step

# Setup: Set a specific step to verify retrieval
test_step_value = 42
tf.summary.experimental.set_step(test_step_value)

if __name__ == '__main__':
    # 1. Eager Execution
    print("Running Eager Execution...")
    try:
        out_eager = get_step_wrapper()
        print(f"Eager Result: {out_eager}")
        eager_success = True
    except Exception as e:
        print(f"Eager Failed: {e}")
        eager_success = False

    # 2. Compiled Execution (tf.function)
    # This is the TensorFlow equivalent of torch.compile
    print("\nRunning Compiled Execution (tf.function)...")
    try:
        compiled_get_step = tf.function(get_step_wrapper)
        out_compiled = compiled_get_step()
        print(f"Compiled Result: {out_compiled}")
        compiled_success = True
    except Exception as e:
        print(f"Compiled Failed: {e}")
        compiled_success = False

    # 3. Verification
    print("\nVerification:")
    if eager_success and compiled_success:
        if out_eager == out_compiled == test_step_value:
            print('Success: Eager and Compiled results match. ')
        else:
            print(f'Failure: Divergence detected! Expected {test_step_value}, '
                  f'Got Eager={out_eager}, Compiled={out_compiled}')
    else:
        print('Failure: One or both modes raised an exception.')

    # Cleanup
    tf.summary.experimental.set_step(None)