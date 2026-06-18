import torch
import tensorflow as tf
import numpy as np

def test_tf_compat_v1_name_scope_conv2d():
    """
    Adapts the PyTorch bug reproduction logic (Conv2d + Compile + Training Loop)
    to TensorFlow using the tf.compat.v1.name_scope API.
    
    Original Bug: torch.compile fails with weight_norm and nn.Conv2d when d > 64.
    Adaptation: Uses tf.compat.v1.name_scope to scope the model definition and 
                tf.function (JIT compile) for the training step.
    """
    # Parameters from the original bug report
    d = 65  # Output channels > 64 to trigger the specific dimension condition
    batch_size = 1
    in_channels = 2
    height, width = 32, 32

    # Input data (NHWC format for TensorFlow)
    x = np.random.randn(batch_size, height, width, in_channels).astype(np.float32)

    # Use the target API: tf.compat.v1.name_scope
    # This wraps the model definition, organizing the created variables/ops
    with tf.compat.v1.name_scope("bug_reproduction_scope"):
        inputs = tf.keras.Input(shape=(height, width, in_channels))
        
        # Conv2D with filters > 64 to match the bug condition
        # Note: TensorFlow uses NHWC, PyTorch uses NCHW
        # Note: Standard TF does not have a direct 'weight_norm' wrapper in core 
        # without tensorflow_addons, so we use a standard Conv2D here to ensure 
        # the test is runnable, focusing on the name_scope API behavior.
        conv = tf.keras.layers.Conv2D(filters=d, kernel_size=2, name="conv2d_layer")
        outputs = conv(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.SGD()

    # Mimic torch.compile using tf.function with JIT compilation enabled
    # This is the semantic equivalent in TensorFlow for optimization/compilation
    @tf.function(jit_compile=True)
    def train_step(inputs):
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(inputs, training=True)
            loss = tf.reduce_mean(predictions)
        
        # Backward pass
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Run the training loop
    # Original bug looped 1000 times, we reduce for a quick test
    try:
        for _ in range(10):
            loss = train_step(x)
            # Assert that loss is a valid tensor (not NaN/Inf) to ensure execution
            tf.debugging.assert_all_finite(loss, message="Loss is NaN or Inf")
        
        print("Test passed: tf.compat.v1.name_scope with Conv2D and JIT compilation executed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    # Ensure eager execution is enabled (default in TF 2.x) but compat.v1 APIs are accessible
    tf.compat.v1.disable_eager_execution() # Optional: strictly use graph mode if desired, but tf.function works in eager too.
    # Re-enabling eager for standard script execution unless graph mode is strictly required for the specific API test.
    # Actually, tf.compat.v1.name_scope works in both. Let's stick to default TF2 behavior.
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution() 
    
    test_tf_compat_v1_name_scope_conv2d()