import tensorflow as tf
import time

# This test case adapts the logic of the Stable Diffusion reproduction script
# to the tf.keras.losses.SparseCategoricalCrossentropy API.
# The original script loads a model, runs inference in a loop, and measures time.
# Here, we instantiate the loss function, run calculations in a loop, and verify the result.

def test_sparse_categorical_crossentropy_performance():
    # Setup: Instantiate the API (analogous to StableDiffusionPipeline.from_pretrained)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()

    # Setup: Define inputs (analogous to the prompt)
    # Using the example data from the API documentation
    y_true = [1, 2]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

    # Convert to tensors
    y_true_tensor = tf.constant(y_true)
    y_pred_tensor = tf.constant(y_pred)

    # Check for GPU availability (analogous to .to("cuda"))
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    with tf.device(device):
        # Execution: Loop 10 times (analogous to the image generation loop)
        start = time.time()
        loss_value = None
        for i in range(10):
            loss_value = scce(y_true_tensor, y_pred_tensor)
        duration = time.time() - start
        print(f"Time taken: {duration}")

        # Verification: Check the result (analogous to saving the image)
        # The expected value is 1.177 based on the API documentation
        expected_loss = 1.177
        # Note: Floating point precision might vary slightly
        assert abs(loss_value.numpy() - expected_loss) < 1e-5, \
            f"Expected loss {expected_loss}, but got {loss_value.numpy()}"

if __name__ == "__main__":
    test_sparse_categorical_crossentropy_performance()