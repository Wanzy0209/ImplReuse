import torch
import torch.nn as nn
import os

def test_dynamo_pixel_shuffle_onnx_export():
    """
    Test case to reproduce the issue where the Dynamo-based ONNX exporter
    adds unnecessary operations around PixelShuffle.
    """
    
    # Handle missing onnx dependency gracefully
    try:
        import onnx
    except ImportError:
        print("Skipping test: 'onnx' module is not installed.")
        return
    
    class Model(nn.Module):
        def __init__(self, kernel_size=3, upscale_factor=2):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, padding="same")
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        def forward(self, input):
            x = self.conv(input)
            x = self.pixel_shuffle(x)
            return x

    model = Model()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.ones((1, 1, 128, 128))
    x = x.to(device)

    onnx_path = "dynamo_pixel_shuffle.onnx"
    
    # Clean up previous file if exists
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

    try:
        # Export the model using Dynamo
        torch.onnx.export(
            model,
            x,
            onnx_path,
            dynamo=True,
            external_data=False,
            opset_version=20,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"input": {0: "batch", 2: "width", 3: "height"}},
        )

        # Verify the model is valid ONNX
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Inspect the graph structure
        graph = onnx_model.graph
        node_ops = [node.op_type for node in graph.node]
        
        # PixelShuffle should map to DepthToSpace in ONNX
        assert "DepthToSpace" in node_ops, "Expected DepthToSpace node in the exported graph"
        
        # Note: The bug report mentions "unnecessary operations" being added.
        # This test ensures the export succeeds and the core operation is present.
        # Further assertions could be added here to check for specific redundant nodes
        # if the exact nature of the unnecessary ops is known.
        
        print("Export successful. Graph operations:", node_ops)

    finally:
        # Clean up
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

if __name__ == "__main__":
    test_dynamo_pixel_shuffle_onnx_export()