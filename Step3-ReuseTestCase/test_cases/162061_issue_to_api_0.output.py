import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

class Model(nn.Module):
    def __init__(self, kernel_size=3, upscale_factor=2):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        x = self.conv(input)
        x = self.pixel_shuffle(x)
        return x

def test_pixel_shuffle_onnx_export():
    """
    Test case to verify ONNX export of a model with PixelShuffle.
    Leverages the pattern from tf.keras.metrics.sparse_top_k_categorical_accuracy
    to explicitly verify tensor ranks (dimensions) in the exported graph.
    """
    model = Model()
    model.eval()
    
    # Input tensor
    x = torch.ones((1, 1, 128, 128))
    
    # 1. PyTorch Inference
    with torch.no_grad():
        torch_out = model(x)

    # 2. Export to ONNX using Dynamo
    torch.onnx.export(
        model,
        x,
        "dynamo.onnx",
        dynamo=True,
        opset_version=20,
        input_names=["input"],
        output_names=["output"],
    )

    # 3. Load and check ONNX model
    onnx_model = onnx.load("dynamo.onnx")
    onnx.checker.check_model(onnx_model)

    # 4. Leverage Similar API Pattern: Check Ranks
    # The similar API (tf.keras.metrics.sparse_top_k_categorical_accuracy) 
    # explicitly checks tensor ranks (y_pred_rank, y_true_rank) via .shape.ndims.
    # We apply this pattern here to ensure the exported graph maintains expected dimensionality.
    graph = onnx_model.graph
    
    # Check input rank
    input_shape = graph.input[0].type.tensor_type.shape.dim
    input_rank = len(input_shape)
    assert input_rank == 4, f"Expected input rank 4, got {input_rank}"

    # Check output rank
    output_shape = graph.output[0].type.tensor_type.shape.dim
    output_rank = len(output_shape)
    assert output_rank == 4, f"Expected output rank 4, got {output_rank}"

    # 5. Verify Numerical Accuracy
    # Ensure the exported model produces the same results as the PyTorch model
    sess = ort.InferenceSession("dynamo.onnx")
    onnx_out = sess.run(None, {"input": x.numpy()})[0]
    
    np.testing.assert_allclose(torch_out.detach().numpy(), onnx_out, rtol=1e-3, atol=1e-5)

if __name__ == "__main__":
    test_pixel_shuffle_onnx_export()
    print("Test passed.")