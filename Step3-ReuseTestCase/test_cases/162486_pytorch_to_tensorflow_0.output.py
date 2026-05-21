import torch
import tensorflow as tf

# Define the loop body function for tf.raw_ops.For
# The body takes a list of tensors [index, ...state] and returns [next_index, ...next_state]
@tf.function
def loop_body(inputs):
    i = inputs[0]
    state = inputs[1]
    # Simple operation: add index to state
    new_state = state + tf.cast(i, tf.float32)
    return [i + 1, new_state]

def test_bug(device: str = '/CPU:0'):
    # In TensorFlow, we use a context manager to set the device for the scope,
    # analogous to torch.set_default_device.
    with tf.device(device):
        try:
            # Create data tensors (analogous to x, y in PyTorch)
            start = tf.constant(0, dtype=tf.int32)
            limit = tf.constant(10, dtype=tf.int32)
            delta = tf.constant(1, dtype=tf.int32)
            initial_state = tf.constant([0.0, 0.0, 0.0])

            # Execute the operation
            # PyTorch: random_split(dataset, lengths)
            # TensorFlow: tf.raw_ops.For(start, limit, delta, inputs, body)
            
            # We wrap the execution in tf.function to ensure the raw op 
            # correctly handles the python function body.
            @tf.function
            def run_for_loop():
                return tf.raw_ops.For(
                    start=start,
                    limit=limit,
                    delta=delta,
                    inputs=[initial_state],
                    body=loop_body
                )

            outputs = run_for_loop()
            
            # Verify output
            assert outputs is not None
            assert len(outputs) == 2
            # The loop runs 10 times (0 to 9). Sum of 0..9 = 45.
            # Initial state [0,0,0] -> Final state [45, 45, 45]
            expected_state = [45.0, 45.0, 45.0]
            # Check if result matches expected (allowing for float precision)
            assert tf.reduce_all(tf.abs(outputs[1] - expected_state) < 1e-5)

            print(f"Device {device} worked.")

        except Exception as e:
            print(f"Device {device} failed: {e}")
            raise

# Run tests
test_bug(device='/CPU:0') # works

# Check for GPU availability to test the 'cuda' equivalent
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    test_bug(device='/GPU:0') # Test on GPU
else:
    print("No GPU available to test '/GPU:0' behavior.")