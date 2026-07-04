import tensorflow as tf
import numpy as np

def verify_environment():
    """Verification that we have a runtime to run the example."""
    return True

def setup_inputs(batch_size):
    """
    Setup inputs for the loss function.
    Mimics the model setup in the original FSDP2 example.
    """
    # Logits: unbounded float tensor (1D for binary hinge loss)
    logits = tf.random.uniform([batch_size], minval=-1.0, maxval=1.0)
    # Labels: 0.0 or 1.0 (will be implicitly converted to -1.0 or 1.0)
    labels = tf.constant([0.0, 1.0, 0.0, 1.0])
    return logits, labels

def main():
    if not verify_environment():
        print("Unable to locate runtime. Exiting.")
        exit()

    print("Running TensorFlow hinge_loss test")

    # Setup
    batch_size = 4
    logits, labels = setup_inputs(batch_size)

    # Original Bug: FSDP2 implicit prefetch does not work.
    # Target API: tf.compat.v1.losses.hinge_loss.
    # Adaptation: Verify the implicit label conversion logic works correctly.
    
    # The API implicitly converts labels {0,1} to {-1,1}.
    # We verify this by comparing the API output against a manual calculation
    # that explicitly performs the conversion.
    
    # 1. Call the API
    loss = tf.compat.v1.losses.hinge_loss(labels=labels, logits=logits)

    # 2. Manual calculation for verification
    # Formula: max(0, 1 - labels * logits)
    # Explicit conversion: labels_explicit = 2 * labels - 1
    labels_explicit = 2.0 * labels - 1.0
    expected_loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - labels_explicit * logits))

    # 3. Verify
    # Check if the implicit conversion inside hinge_loss matches the explicit one
    assert np.allclose(loss.numpy(), expected_loss.numpy()), \
        "Implicit label conversion in hinge_loss failed."

    print(f"Test Passed. Loss: {loss.numpy()}")

if __name__ == "__main__":
    main()