import torch
import torch.onnx
import onnx
from torchvision.models import resnet50, ResNet50_Weights
import os

def test_onnx_dynamo_conv_bias_shape():
    """
    Test case to verify that torch.onnx.export with dynamo=True produces
    correct Conv bias tensor shapes, matching the classical export path.
    
    This test leverages torch.backends.cuda.is_built to determine the 
    appropriate device for execution.
    """
    
    # Leverage the similar API to check for CUDA availability
    # and set the device accordingly.
    if torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create ResNet50 model with pretrained weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Define output paths
    path_classical = "resnet_classical.onnx"
    path_dynamo = "resnet_dynamo.onnx"

    try:
        # Export using the classical path
        torch.onnx.export(
            model, 
            dummy_input, 
            path_classical,
            dynamo=False,
            opset_version=17,
            input_names=['input'], 
            output_names=['output']
        )

        # Export using dynamo
        torch.onnx.export(
            model, 
            dummy_input, 
            path_dynamo,
            dynamo=True,
            opset_version=17,
            input_names=['input'], 
            output_names=['output']
        )

        # Load the exported models
        model_classical = onnx.load(path_classical)
        model_dynamo = onnx.load(path_dynamo)

        # Helper function to extract Conv bias shapes from the graph
        def get_conv_bias_shapes(onnx_model):
            bias_shapes = {}
            # Create a map of initializer names to their tensors
            initializer_map = {init.name: init for init in onnx_model.graph.initializer}
            
            for node in onnx_model.graph.node:
                if node.op_type == 'Conv':
                    # Bias is typically the 3rd input (index 2) to a Conv node
                    if len(node.input) > 2:
                        bias_name = node.input[2]
                        if bias_name in initializer_map:
                            bias_shapes[node.name] = list(initializer_map[bias_name].dims)
            return bias_shapes

        classical_shapes = get_conv_bias_shapes(model_classical)
        dynamo_shapes = get_conv_bias_shapes(model_dynamo)

        # Assertion to catch the bug described in Issue 166797
        # The bug report specifically mentions incorrect bias shapes for Conv layers
        assert classical_shapes == dynamo_shapes, (
            f"Mismatch found in Conv bias shapes between classical and dynamo exports.\n"
            f"Classical shapes: {classical_shapes}\n"
            f"Dynamo shapes: {dynamo_shapes}"
        )
        
        print("Test passed: Conv bias shapes match between classical and dynamo exports.")

    finally:
        # Clean up generated files
        if os.path.exists(path_classical):
            os.remove(path_classical)
        if os.path.exists(path_dynamo):
            os.remove(path_dynamo)

if __name__ == "__main__":
    test_onnx_dynamo_conv_bias_shape()