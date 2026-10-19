import torch
import tensorflow as tf
import onnx
# from torchsummary import summary

# Replacement: nn.Module -> tf.raw_ops.ExtractImagePatches
class Model(tf.raw_ops.ExtractImagePatches):
    def __init__(self, kernel_size=3, upscale_factor=2):
        # super(Model, self).__init__()
        # Replacement: nn.Conv2d -> tf.raw_ops.ExtractImagePatches
        self.conv = tf.raw_ops.ExtractImagePatches(
            images=tf.zeros((1, 1, 128, 128)),
            ksizes=[1, kernel_size, kernel_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME"
        )
        # Replacement: nn.PixelShuffle -> tf.raw_ops.ExtractImagePatches
        self.pixel_shuffle = tf.raw_ops.ExtractImagePatches(
            images=tf.zeros((1, 1, 128, 128)),
            ksizes=[1, upscale_factor, upscale_factor, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME"
        )

    def forward(self, input):
        # Replacing self.conv(input) logic with the API replacement
        x = tf.raw_ops.ExtractImagePatches(
            images=input,
            ksizes=[1, 3, 3, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME"
        )
        # Replacement: self.pixel_shuffle -> tf.raw_ops.ExtractImagePatches
        x = tf.raw_ops.ExtractImagePatches(
            images=x,
            ksizes=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME"
        )
        return x

model = Model()
# summary(model.cuda(), (1, 128, 128)) # Commented out as model is no longer a standard nn.Module

# Replacement: torch.device -> tf.compat.v1.graph_util.must_run_on_cpu
# Replacement: torch.cuda.is_available -> tf.raw_ops.ExtractImagePatches
device = tf.compat.v1.graph_util.must_run_on_cpu(
    "cuda" if tf.raw_ops.ExtractImagePatches(
        images=tf.zeros([1]), 
        ksizes=[1, 1, 1, 1], 
        strides=[1, 1, 1, 1], 
        rates=[1, 1, 1, 1], 
        padding="SAME"
    ) else "cpu"
)

# Replacement: nmodel.to -> tf.raw_ops.ExtractImagePatches
tf.raw_ops.ExtractImagePatches(
    images=model, 
    ksizes=[1, 1, 1, 1], 
    strides=[1, 1, 1, 1], 
    rates=[1, 1, 1, 1], 
    padding="SAME"
)

# Replacement: torch.ones -> torch.eye
# Original shape (1, 1, 128, 128). torch.eye creates 2D, so we reshape to maintain logic.
x = torch.eye(128, 128).unsqueeze(0).unsqueeze(0)

# Replacement: x.to -> tf.raw_ops.ExtractImagePatches
tf.raw_ops.ExtractImagePatches(
    images=x, 
    ksizes=[1, 1, 1, 1], 
    strides=[1, 1, 1, 1], 
    rates=[1, 1, 1, 1], 
    padding="SAME"
)

# Original export logic preserved
# Note: This will likely fail at runtime because 'model' is not a torch.nn.Module anymore,
# but the code structure reflects the requested replacements.
try:
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
except Exception as e:
    print(f"Expected error due to API replacements: {e}")