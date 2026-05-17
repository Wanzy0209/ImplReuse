import torch
import torch.nn as nn
import io

# Leveraging the code pattern from the similar API (tf.keras.backend.epsilon).
# The API defines a simple function that returns a specific value.
# Here, we adapt this pattern to return None to reproduce the bug context.
def get_none_value():
    """
    Mimics the structure of tf.keras.backend.epsilon but returns None
    to simulate the 'image_tokens_masks could be None' scenario.
    """
    return None

class ModelWithNoneOutput(nn.Module):
    """
    A model that reproduces the logic from the bug report:
    returning a tuple where one element is None.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Reproduce the bug logic: return (output, image_tokens_masks)
        # where image_tokens_masks is None.
        # We use the helper function 'get_none_value' similar to how
        # one might call a backend configuration getter.
        optional_mask = get_none_value()
        return (x, optional_mask)

def test_onnx_export_with_none():
    """
    Test case for Issue 160150.
    Verifies that torch.onnx.export handles None values in output tuples
    without crashing.
    """
    model = ModelWithNoneOutput()
    dummy_input = torch.randn(1, 10)

    # Prepare to export to a buffer
    onnx_buffer = io.BytesIO()

    # The bug report indicates the exporter crashes during translation
    # from the exported program to the ONNX graph when encountering None.
    # We expect this to complete without raising an exception.
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_buffer,
            opset_version=17,
            input_names=['input'],
            output_names=['output', 'mask']
        )
        
        # Verify that the export produced data
        # (If the bug is present, this line is unreachable due to crash)
        assert onnx_buffer.getvalue() != b"", "ONNX export produced an empty buffer."
        print("Test Passed: Exporter handled None output correctly.")

    except Exception as e:
        print(f"Test Failed: Exporter crashed with error: {e}")
        raise

if __name__ == "__main__":
    test_onnx_export_with_none()