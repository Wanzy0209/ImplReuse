#!/usr/bin/env python3

"""
Test case to reproduce the Torch ONNX Dynamo Export bug where bias tensor shapes
are incorrect for Conv layers.
"""

import torch
import torch.onnx
from torchvision.models import resnet50, ResNet50_Weights
import onnx
from onnx import numpy_helper

def test_resnet50_dynamo_export_bias_shape():
    """
    Test that torch.onnx.export with dynamo=True produces correct bias shapes
    for Conv layers in ResNet50.
    
    The bug report indicates that node_Conv_649 (and potentially others) 
    have incorrect bias shapes when exported with dynamo=True.
    """
    # Create ResNet50 model with pretrained weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX using dynamo
    onnx_path = "resnet50_dynamo.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=21,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=True,
        fallback=False,
    )

    # Load the exported ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Verify the model structure is valid
    onnx.checker.check_model(onnx_model)

    # Create a map of initializers (weights/biases) for easy lookup
    initializer_map = {init.name: init for init in onnx_model.graph.initializer}

    # Iterate through nodes to find Conv layers and check bias shapes
    incorrect_biases = []
    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            # Conv inputs are typically: [input, weight, bias]
            # Bias is optional, so check if it exists
            if len(node.input) > 2:
                bias_name = node.input[2]
                if bias_name in initializer_map:
                    bias_tensor = initializer_map[bias_name]
                    bias_shape = list(bias_tensor.dims)
                    
                    # In ONNX, Conv bias should be 1D (shape: [out_channels]).
                    # The bug report implies the shape is "obviously incorrect",
                    # often meaning it has extra dimensions (e.g., 4D).
                    if len(bias_shape) != 1:
                        incorrect_biases.append({
                            'node_name': node.name,
                            'bias_name': bias_name,
                            'shape': bias_shape
                        })

    # Assert that no incorrect biases were found
    assert len(incorrect_biases) == 0, (
        f"Found {len(incorrect_biases)} Conv nodes with incorrect bias shapes. "
        f"Expected 1D shape. Details: {incorrect_biases}"
    )

    print("Test passed: All Conv bias shapes are correct.")

if __name__ == "__main__":
    test_resnet50_dynamo_export_bias_shape()