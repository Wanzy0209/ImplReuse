import torch
import tensorflow as tf

def test_vdot_behavior():
    # Adapt the input data from the PyTorch example
    # Original PyTorch input: torch.tensor([1.0, 2.0, 3.0])
    # We use these values for both inputs of vdot to test the operation
    input_a = tf.constant([1.0, 2.0, 3.0])
    input_b = tf.constant([1.0, 2.0, 3.0])

    # Call the similar API: tf.keras.ops.vdot
    # vdot computes the dot product of the flattened inputs
    # This adapts the logic of applying an operator to the specific input data
    out_tensor = tf.keras.ops.vdot(input_a, input_b)

    print("Input A:", input_a)
    print("Input B:", input_b)
    print("Output tensor:", out_tensor)

    # Verify the result (1*1 + 2*2 + 3*3 = 14)
    assert out_tensor == 14.0, f"Expected 14.0, got {out_tensor}"

if __name__ == "__main__":
    test_vdot_behavior()