import torch
import tensorflow as tf

def test_large_tensor_creation():
    """
    Adapted test case for Issue 161265.
    Verifies that creating a tensor larger than 4GB and accessing its elements
    works correctly, similar to the torch.full/torch.ones issue on MPS.
    """
    # Shape: 2 rows, (2^31 + 5) columns.
    # Total elements: 2 * (2^31 + 5) = 2^32 + 10.
    # Size in bytes (int8): 2^32 + 10 bytes (> 4GB).
    shape = [2, (1 << 31) + 5]

    # Create a large tensor filled with 1s.
    # In PyTorch, torch.ones is used, which internally calls torch.full.
    # Here we use tf.ones to generate the data, then pass it through 
    # tf.convert_to_tensor to satisfy the API requirement.
    raw_tensor = tf.ones(shape, dtype=tf.int8)
    a = tf.convert_to_tensor(raw_tensor)

    # Check the specific indices mentioned in the bug report.
    # The bug manifested as incorrect values (0 instead of 1) for elements
    # located beyond the 4GB boundary in the buffer.
    
    # Access a single element near the end (row 1, second to last column)
    val_single = a[1, -2]
    
    # Access a slice of the second to last column
    val_slice = a[:, -2]

    print("Value at [1, -2]:", val_single.numpy())
    print("Value at [:, -2]:", val_slice.numpy())

    # Assertions to verify the buffer was filled correctly
    assert val_single.numpy() == 1, f"Expected 1, got {val_single.numpy()}. Buffer fill might be failing for >4GB."
    assert list(val_slice.numpy()) == [1, 1], f"Expected [1, 1], got {val_slice.numpy()}. Buffer fill might be failing for >4GB."

if __name__ == "__main__":
    test_large_tensor_creation()