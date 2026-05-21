import torch
import tensorflow as tf
from tensorflow.keras import layers, Model

# Check for GPU availability (similar to torch.backends.mps.is_available())
# This mimics the environment check in the original bug report.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        print(e)

# Wrapper over the custom TF resize bicubic kernel.
# Adapted from MPSSoftshrink in the original report.
class TFResizeBicubic(layers.Layer):
    def __init__(self, size=(64, 64), align_corners=False, half_pixel_centers=False):
        super(TFResizeBicubic, self).__init__()
        self.size = size
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

    def call(self, inputs):
        return tf.compat.v1.image.resize_bicubic(
            images=inputs,
            size=self.size,
            align_corners=self.align_corners,
            half_pixel_centers=self.half_pixel_centers
        )

# Wrapper over the Sequential layer, using the custom TF resize bicubic implementation.
# Adapted from CustomMPSSoftshrinkModel in the original report.
class CustomTFResizeModel(Model):
    def __init__(
        self,
        input_shape=(28, 28, 1),
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        output_size: int = 10,
    ):
        super(CustomTFResizeModel, self).__init__()

        self.model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(conv1_filters, (3, 3), activation='relu'),
            TFResizeBicubic(size=(14, 14)), # Custom Op Layer 1
            layers.Conv2D(conv2_filters, (3, 3), activation='relu'),
            TFResizeBicubic(size=(7, 7)),  # Custom Op Layer 2
            layers.Flatten(),
            layers.Dense(output_size)
        ])

    def call(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Create dummy input data (Batch, Height, Width, Channels)
    # Mimicking the input flow of the original PyTorch code
    input_data = tf.random.normal((1, 28, 28, 1))

    model = CustomTFResizeModel()
    
    # Run the model to verify behavior
    output = model(input_data)
    
    print("Test passed. Output shape:", output.shape)
    assert output.shape == (1, 10)