import torch
import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    # Handle environment dependency error (e.g., GLIBCXX version mismatch)
    print(f"Skipping test: TensorFlow import failed due to environment issues ({e}).")
else:
    # Instantiate MobileNetV2 without pre-trained weights
    # Equivalent to torchvision.models.mobilenet_v2(weights=None)
    model = tf.keras.applications.MobileNetV2(weights=None)

    # Create a dummy input tensor
    # Note: TensorFlow uses channels_last (NHWC) format by default, unlike PyTorch's NCHW
    x = tf.random.uniform((1, 224, 224, 3))

    # Clone the model structure
    # tf.keras.models.clone_model creates a new model with newly initialized weights.
    # To verify the cloned model behaves identically to the original (similar to torch.export),
    # we explicitly copy the weights from the original model to the clone.
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())

    # Run inference on both the original and the cloned model
    y_original = model(x)
    y_cloned = cloned_model(x)

    # Assert that the outputs are numerically close
    # Equivalent to torch.testing.assert_close
    np.testing.assert_allclose(y_original.numpy(), y_cloned.numpy(), rtol=1e-5, atol=1e-5)