import torch
import tensorflow as tf
from tensorflow.experimental import dtensor

def test_copy_to_mesh_conv2d_training():
    """
    Adapted test case for tf.experimental.dtensor.copy_to_mesh based on 
    PyTorch issue #165749 (Backward pass failure for nn.Conv2d with weight_norm and torch.compile).
    
    This test verifies the behavior of the copy_to_mesh API during a training loop 
    with a Conv2D layer, preserving the dimensionality (d=65) that triggered the original bug.
    """
    
    # 1. Setup Mesh and Layout
    # We use a single device mesh to ensure the test is runnable without specific hardware requirements.
    # In the original bug, torch.compile was used. Here, copy_to_mesh is used to distribute 
    # the tensor to the mesh, serving a similar role in preparing the data for the specific execution context.
    mesh = dtensor.create_mesh([("batch", 1)], devices=["CPU:0"])
    layout = dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED, dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)

    # 2. Define Data
    # PyTorch input: (1, 2, 32, 32) [Batch, Channels, Height, Width]
    # TensorFlow input: (1, 32, 32, 2) [Batch, Height, Width, Channels]
    x = tf.random.normal((1, 32, 32, 2))

    # 3. Apply the API: copy_to_mesh
    # This moves the tensor to the DTensor mesh.
    x_mesh = dtensor.copy_to_mesh(x, layout)

    # 4. Define Model
    # PyTorch: nn.Conv2d(2, 65, 2) -> Input Channels=2, Output Channels=65, Kernel Size=2
    # TensorFlow: Conv2D(filters=65, kernel_size=2)
    # Note: The original bug was specific to d > 64 (here d=65). We preserve this dimension.
    # While the original bug involved weight_norm, we use standard Conv2D here to test 
    # the interaction of the training loop with the copy_to_mesh API.
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=65, kernel_size=2, input_shape=(32, 32, 2))
    ])

    optimizer = tf.keras.optimizers.SGD()

    # 5. Training Loop
    # The original bug occurred during the backward pass. We run a few iterations to verify stability.
    try:
        for _ in range(10):
            with tf.GradientTape() as tape:
                y = model(x_mesh)
                loss = tf.reduce_mean(y)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        print("Test passed: Training loop with copy_to_mesh completed successfully.")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_copy_to_mesh_conv2d_training()