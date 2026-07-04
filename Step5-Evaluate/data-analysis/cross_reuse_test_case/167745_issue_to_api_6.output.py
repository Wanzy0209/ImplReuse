import torch
import tensorflow as tf

def test_nested_custom_gradients_inline():
    """
    This test mirrors the failing patterns in the PyTorch MemPool bug (Issue 167745).
    Specifically, it tests the usage of temporary objects (lambdas) passed to the API
    (custom_gradient) and nested contexts (GradientTape/Operations).

    PyTorch Pattern (Test 2):
        with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
            with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool2)):
                ...

    TensorFlow Adaptation:
    We define custom gradient operations using temporary lambda functions (analogous to 
    the temporary MemPool objects) and nest their usage within a GradientTape context.
    """

    # Define an inner custom gradient operation using a temporary lambda
    # This mirrors: torch.cuda.MemPool(pool2)
    op_inner = tf.keras.ops.custom_gradient(
        lambda x: (
            x + 1.0,  # Forward pass: y = x + 1
            lambda dy: dy  # Gradient pass: dy/dx = 1
        )
    )

    # Define an outer custom gradient operation using a temporary lambda
    # This mirrors: torch.cuda.MemPool(pool1)
    op_outer = tf.keras.ops.custom_gradient(
        lambda x: (
            op_inner(x) * 2.0,  # Forward pass: y = (x + 1) * 2
            lambda dy: dy * 2.0  # Gradient pass: dy/dx = 2
        )
    )

    x = tf.constant(3.0)

    # Use the operations within a context
    # This mirrors the 'with torch.cuda.use_mem_pool(...):' block
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = op_outer(x)

    # Verify forward pass: (3.0 + 1.0) * 2.0 = 8.0
    assert y == 8.0, f"Expected forward result 8.0, got {y}"

    # Verify backward pass (gradient)
    # d/dx [(x + 1) * 2] = 2
    grad = tape.gradient(y, x)
    assert grad == 2.0, f"Expected gradient 2.0, got {grad}"

if __name__ == "__main__":
    test_nested_custom_gradients_inline()
    print("Test passed.")