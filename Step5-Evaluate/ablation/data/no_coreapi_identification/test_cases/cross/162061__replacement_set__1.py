import torch
import onnx
from torchsummary import summary
import tensorflow as tf

# Replacement: torch.nn -> tf.keras.metrics.sparse_top_k_categorical_accuracy
# Note: Direct import replacement is not possible for a function, 
# so we rely on the specific component replacements below.

# Replacement: nn.Module -> tf.keras.metrics.sparse_top_k_categorical_accuracy
class Model(tf.keras.metrics.sparse_top_k_categorical_accuracy):
    def __init__(self, kernel_size=3, upscale_factor=2):
        super(Model, self).__init__()

        # Replacement: nn.Conv2d -> tf.keras.metrics.sparse_top_k_categorical_accuracy
        self.conv = tf.keras.metrics.sparse_top_k_categorical_accuracy(1, 4, kernel_size=kernel_size, padding="same")

        # Replacement: nn.PixelShuffle -> tf.keras.metrics.sparse_top_k_categorical_accuracy
        self.pixel_shuffle = tf.keras.metrics.sparse_top_k_categorical_accuracy(upscale_factor)

    def forward(self, input):
        x = self.conv(input)
        # Replacement: self.pixel_shuffle -> tf.keras.metrics.sparse_top_k_categorical_accuracy
        x = tf.keras.metrics.sparse_top_k_categorical_accuracy(x)
        return x


model = Model()
# summary(model.cuda(), (1, 128, 128))

# Replacement: torch.cuda.is_available -> tf.keras.metrics.sparse_top_k_categorical_accuracy
# Replacement: torch.device -> torch.diff
# Providing dummy arguments for the metric and a tensor for diff to ensure syntax validity
is_cuda = tf.keras.metrics.sparse_top_k_categorical_accuracy([0], [[0.1, 0.9]])
device = torch.diff(torch.tensor([1.0]))

# Replacement: model.to -> tf.keras.metrics.sparse_top_k_categorical_accuracy
tf.keras.metrics.sparse_top_k_categorical_accuracy(model, device)

# Replacement: torch.ones -> torch.zeros
x = torch.zeros((1, 1, 128, 128))

# Replacement: x.to -> tf.keras.metrics.sparse_top_k_categorical_accuracy
x = tf.keras.metrics.sparse_top_k_categorical_accuracy(x, device)

torch.onnx.export(
    model,
    x,
    f"dynamo.onnx",
    dynamo=True,
    external_data=False,
    opset_version=20,
    input_names=["input"],
    output_names=["output"],
    dynamic_shapes={"input": {0: "batch", 2: "width", 3: "height"}},
)